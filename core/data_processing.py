from function.utils import (
    distribute_batches_equally,
    convert_distributed_data_into_numpy,
    poison_data,
    generate_data_loaders_from_distributed_dataset,
    filter_data_loader_by_label
)
import numpy as np
from collections import defaultdict
import json
import os
from datetime import datetime


def generate_hybrid_client_dataset(args, data_loader, num_clients, seen_per_client=5, alpha=0.5, assign_seed=0, force_seen_sets=None):
    """
    Hybrid partition:
      - deterministically choose seen classes per client (always include class 0)
      - for each class, split its samples among the clients that are allowed to see it using Dirichlet(alpha)

    Returns: distributed_dataset: list of (X, Y) arrays per client, seen_sets: list of sets
    """
    # Use a local RNG for deterministic behavior across runs when assign_seed is fixed
    rng = np.random.RandomState(assign_seed)

    # collect all samples from data_loader into arrays
    X_all = np.array([tensor.numpy() for batch in data_loader for tensor in batch[0]])
    Y_all = np.array([tensor.numpy() for batch in data_loader for tensor in batch[1]]).astype(int)

    classes = sorted(np.unique(Y_all).tolist())
    attack_classes = [c for c in classes if c != 0]

    # decide seen-sets per client
    if force_seen_sets is not None:
        seen_sets = [set(s) for s in force_seen_sets]
    else:
        seen_sets = []
        for i in range(num_clients):
            k = max(0, seen_per_client - 1)
            # if k covers all attack classes, just take all
            if k >= len(attack_classes):
                chosen_attacks = attack_classes
            else:
                k_eff = min(k, len(attack_classes))
                chosen_attacks = list(rng.choice(attack_classes, size=k_eff, replace=False)) if k_eff > 0 else []
            chosen = [0] + [int(c) for c in chosen_attacks]
            seen_sets.append(set(chosen))

    # ensure coverage: every attack class seen by at least one client
    for c in attack_classes:
        if not any(c in s for s in seen_sets):
            j = np.random.randint(0, num_clients)
            seen_sets[j].add(c)

    # indices per class
    idx_by_class = {c: np.where(Y_all == c)[0].tolist() for c in classes}
    for c in idx_by_class:
        # shuffle in-place deterministically
        rng.shuffle(idx_by_class[c])

    # allocate per-class using Dirichlet across candidate clients
    clients_idx = {i: [] for i in range(num_clients)}
    for c, idxs in idx_by_class.items():
        candidate_clients = [i for i in range(num_clients) if int(c) in seen_sets[i]]
        if len(candidate_clients) == 0:
            # leave for global test (not assigned to any client)
            continue
        if len(idxs) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(len(candidate_clients)))
        counts = (proportions * len(idxs)).astype(int)
        remainder = len(idxs) - counts.sum()
        for r in range(remainder):
            counts[r % len(counts)] += 1
        start = 0
        for j, client_id in enumerate(candidate_clients):
            cnt = counts[j]
            if cnt > 0:
                sel = idxs[start:start+cnt]
                clients_idx[client_id].extend(sel)
                start += cnt

    # build distributed dataset list
    distributed_dataset = []
    for i in range(num_clients):
        idxs = np.array(clients_idx[i], dtype=int)
        if idxs.size == 0:
            Xc = np.zeros((0, X_all.shape[1]), dtype=X_all.dtype)
            Yc = np.zeros((0,), dtype=Y_all.dtype)
        else:
            Xc = X_all[idxs]
            Yc = Y_all[idxs].astype(int)
        distributed_dataset.append((Xc, Yc))

    return distributed_dataset, seen_sets

def client_data_process(args, data_loader, poisoned_workers, mal_data_loader, batch_size, poison=True, data_stage="train"):
    # Support new hybrid partition strategy (deterministic class-assignment + Dirichlet within classes)
    if getattr(args, "partition_strategy", "original") == "hybrid":
        args.logger.info("Using hybrid partition strategy: seen_per_client={}, dir_alpha={}, seed={}",
                         args.seen_per_client, args.dir_alpha, args.assign_seed)
        # generate_hybrid_client_dataset expects the full data_loader (will iterate it)
        # For test stage: make each client see all classes (0 + all attacks)
        force_seen_sets = None
        if data_stage == "test":
            # build full class list from incoming loader
            Y_all_tmp = np.array([tensor.numpy() for batch in data_loader for tensor in batch[1]]).astype(int)
            classes_tmp = sorted(np.unique(Y_all_tmp).tolist())
            force_seen_sets = [set(int(c) for c in classes_tmp) for _ in range(args.num_workers)]

        distributed_dataset, seen_sets = generate_hybrid_client_dataset(
            args,
            data_loader,
            args.num_workers,
            seen_per_client=args.seen_per_client,
            alpha=args.dir_alpha,
            assign_seed=args.assign_seed,
            force_seen_sets=force_seen_sets,
        )

        # Log per-client counts per class
        for i, (Xc, Yc) in enumerate(distributed_dataset):
            # build counts dict
            unique, counts = np.unique(Yc, return_counts=True) if Yc.size > 0 else ([], [])
            counts_dict = {int(u): int(c) for u, c in zip(unique, counts)}
            args.logger.info("Client {}: total_samples={}, class_counts={}", i, Yc.shape[0], counts_dict)
        # also log seen_sets mapping
        args.logger.info("Seen classes per client: {}", [sorted(list(s)) for s in seen_sets])
        # persist partition metadata for experiment provenance
        # build client class counts
        client_counts = []
        for i, (Xc, Yc) in enumerate(distributed_dataset):
            unique, counts = np.unique(Yc, return_counts=True) if Yc.size > 0 else ([], [])
            counts_dict = {int(u): int(c) for u, c in zip(unique, counts)}
            client_counts.append(counts_dict)

        meta = {
            "dataset": getattr(args, "dataset", "unknown"),
            "partition_strategy": getattr(args, "partition_strategy", "original"),
            "seen_per_client": getattr(args, "seen_per_client", None),
            "dir_alpha": getattr(args, "dir_alpha", None),
            "assign_seed": getattr(args, "assign_seed", None),
            "num_clients": args.num_workers,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "seen_sets": [[int(x) for x in sorted(list(s))] for s in seen_sets],
            "client_class_counts": client_counts,
        }
        # store meta on args for later steps (experiment runner can read it)
        if not hasattr(args, 'partition_meta_history'):
            args.partition_meta_history = []
        args.partition_meta_history.append(meta)
        # track by stage to use train seen_sets during test/visualize
        if data_stage == "train":
            args.train_partition_meta = meta
        elif data_stage == "val":
            args.val_partition_meta = meta
        elif data_stage == "test":
            args.test_partition_meta = meta
        # keep last_partition_meta pointing to train if available else current
        if hasattr(args, 'train_partition_meta'):
            args.last_partition_meta = args.train_partition_meta
        else:
            args.last_partition_meta = meta
    else:
        distributed_dataset = distribute_batches_equally(data_loader, args.num_workers)
        distributed_dataset = convert_distributed_data_into_numpy(distributed_dataset)

    if poison:
        distributed_dataset = poison_data(
            args.logger,
            distributed_dataset,
            args.num_workers,
            poisoned_workers,
            args.noise_type,
            mal_data_loader,
            args.replacement_ratio,
            args.attack_std_noise,
        )

    data_loaders = generate_data_loaders_from_distributed_dataset(distributed_dataset, batch_size)
    return data_loaders
