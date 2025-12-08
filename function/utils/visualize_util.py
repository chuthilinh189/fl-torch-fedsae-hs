import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
import torch
from sklearn.preprocessing import StandardScaler
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

def visualize_parameters(epoch, clients, poisoned_workers, args):
    client_weights = []
    client_labels = []

    for i, client in enumerate(clients):
        weights = client.get_nn_parameters()
        flat_weights = np.concatenate([w.flatten() for w in weights.values()])
        client_weights.append(flat_weights)
        client_labels.append(1 if i in poisoned_workers else 0)

    client_weights = np.array(client_weights)

    perplexity = min(len(client_weights) - 1, 5)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_weights = tsne.fit_transform(client_weights)

    plt.figure(figsize=(8, 8))
    for i, label in enumerate(client_labels):
        color = 'red' if label == 1 else 'blue'
        marker = 'x' if label == 1 else 'o'
        plt.scatter(reduced_weights[i, 0], reduced_weights[i, 1], c=color, marker=marker)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f"t-SNE Visualization of Client Weights (Epoch {epoch})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    out_dir = os.path.join("visual", args.model_type if hasattr(args, 'model_type') else 'weights')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}_client_weights_tsne.png"))
    plt.close()


def log_re_distributions_from_lists(list_re, raw_labels, out_dir, epoch, client_id=None, normalize_per_client=False):
    """
    Save RE per-sample grouped by attack type and draw simple plots (histogram, boxplot).

    Returns (summary_dict, grouped_dict)
    summary_dict: attack_id -> {count, mean, std}
    grouped_dict: attack_id -> list of RE floats
    """
    os.makedirs(out_dir, exist_ok=True)

    re_arr = np.array(list_re, dtype=float)
    labels_arr = np.array(raw_labels, dtype=int)

    if normalize_per_client:
        mu, sigma = re_arr.mean(), re_arr.std() if re_arr.std() > 0 else 1.0
        re_arr = (re_arr - mu) / sigma

    grouped = defaultdict(list)
    for r, l in zip(re_arr.tolist(), labels_arr.tolist()):
        grouped[l].append(r)

    summary = {}
    fname_base = f"epoch{epoch}"
    if client_id is not None:
        fname_base += f"_client{client_id}"

    return summary, grouped


def collect_re_and_labels_from_client(client, normalize=False):
    """
    Collect per-sample reconstruction error and raw labels from a client's test_data_loader.

    Returns (list_re, raw_labels)
    """
    client.net.eval()
    list_re = []
    raw_labels = []
    device = client.device
    with torch.no_grad():
        for input, label in client.test_data_loader:
            input = input.to(device)
            label = label.to(device)
            encode, decode = client.net(input)
            # per-sample MSE
            per_sample_re = ((decode - input) ** 2).mean(dim=1).cpu().numpy()
            list_re.extend(per_sample_re.tolist())
            raw_labels.extend(label.cpu().numpy().astype(int).tolist())

    if normalize and len(list_re) > 0:
        arr = np.array(list_re)
        arr = (arr - arr.mean()) / (arr.std() if arr.std() > 0 else 1)
        list_re = arr.tolist()

    return list_re, raw_labels


def collect_z_norms_and_labels_from_client(client, normalize=False):
    """
    Collect per-sample latent vector norms (L2) and raw labels from a client's test_data_loader.

    Returns (list_z_norms, raw_labels)
    """
    client.net.eval()
    list_z = []
    raw_labels = []
    device = client.device
    with torch.no_grad():
        for input, label in client.test_data_loader:
            input = input.to(device)
            label = label.to(device)
            encode, decode = client.net(input)
            # per-sample latent L2 norm
            per_sample_z_norm = torch.norm(encode.view(encode.size(0), -1), p=2, dim=1).cpu().numpy()
            list_z.extend(per_sample_z_norm.tolist())
            raw_labels.extend(label.cpu().numpy().astype(int).tolist())

    if normalize and len(list_z) > 0:
        arr = np.array(list_z)
        arr = (arr - arr.mean()) / (arr.std() if arr.std() > 0 else 1)
        list_z = arr.tolist()

    return list_z, raw_labels


def log_z_boxplots_from_lists(list_z, raw_labels, out_dir, epoch, client_id=None):
    """
    For each attack type, create a combined boxplot comparing benign (label 0) vs that attack's latent-norm distribution.
    Also write per-attack CSVs of z-norms. Produces per-client files when client_id is provided, and global when None.
    """
    os.makedirs(out_dir, exist_ok=True)

    z_arr = np.array(list_z, dtype=float)
    labels_arr = np.array(raw_labels, dtype=int)

    fname_base = f"epoch{epoch}"
    if client_id is not None:
        fname_base += f"_client{client_id}"

    # group by label
    grouped = defaultdict(list)
    for z, l in zip(z_arr.tolist(), labels_arr.tolist()):
        grouped[l].append(z)

    # overall per-attack box (for visualization across attacks show distribution of z-norm per attack)
    atk_ids = sorted([k for k in grouped.keys()])
    if len(atk_ids) > 1:
        data = [grouped[a] for a in atk_ids]
        plt.figure(figsize=(max(6, len(atk_ids) * 0.6), 4))
        plt.boxplot(data)
        plt.xticks(range(1, len(atk_ids) + 1), [str(a) for a in atk_ids])
        plt.title(f"Latent L2-norm per attack type - epoch {epoch}")
        plt.xlabel("attack id")
        plt.ylabel("L2 norm of latent z")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname_base}_per_attack_z_box.png"))
        plt.close()

    return grouped


def compute_auc_per_attack_from_flat(y_true, re_scores):
    """
    Compute one-vs-rest AUC per attack id using benign (label 0) as negatives.
    Returns dict attack_id -> auc (float) or None if not computable.
    """
    y_true = np.array(y_true)
    re_scores = np.array(re_scores)
    unique_labels = sorted(np.unique(y_true))
    aucs = {}
    for atk in unique_labels:
        if atk == 0:
            continue
        pos_mask = (y_true == atk)
        neg_mask = (y_true == 0)
        if neg_mask.sum() == 0 or pos_mask.sum() == 0:
            aucs[atk] = None
            continue
        y_bin = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
        scores = np.concatenate([re_scores[pos_mask], re_scores[neg_mask]])
        try:
            aucs[atk] = float(roc_auc_score(y_bin, scores))
        except Exception:
            aucs[atk] = None
    return aucs


def collect_latents_and_labels_from_client(client, max_samples_per_class=None):
    """
    Collect per-sample latent vectors and raw labels from a client's test_data_loader.

    Returns (latents_numpy_array, labels_list) where latents shape = (N, D).
    If max_samples_per_class is provided, this will sample up to that many examples per label to keep embeddings fast.
    """
    client.net.eval()
    latents = []
    labels = []
    device = client.device
    with torch.no_grad():
        for input, label in client.test_data_loader:
            input = input.to(device)
            label = label.to(device)
            encode, decode = client.net(input)
            enc_np = encode.view(encode.size(0), -1).cpu().numpy()
            latents.append(enc_np)
            labels.extend(label.cpu().numpy().astype(int).tolist())

    if len(latents) == 0:
        return np.zeros((0, 0)), []

    latents = np.concatenate(latents, axis=0)

    # optional balanced sampling per class to limit size
    if max_samples_per_class is not None and max_samples_per_class > 0:
        kept_idx = []
        labels_arr = np.array(labels)
        for lbl in np.unique(labels_arr):
            idxs = np.where(labels_arr == lbl)[0]
            if len(idxs) > max_samples_per_class:
                chosen = np.random.choice(idxs, size=max_samples_per_class, replace=False)
            else:
                chosen = idxs
            kept_idx.extend(chosen.tolist())
        kept_idx = sorted(kept_idx)
        latents = latents[kept_idx]
        labels = labels_arr[kept_idx].tolist()

    return latents, labels


def plot_latent_embedding(latents, labels, out_dir, epoch, client_id=None, proto_z0=None, proto_z1=None, method="tsne", max_points=1000, random_state=42):
    """
    Create a 2D scatter of latent vectors using t-SNE or UMAP, color by label and overlay prototypes when provided.

    - latents: numpy array shape (N, D)
    - labels: list/array shape (N,)
    - out_dir: directory to save images
    - epoch, client_id: used for filename
    - proto_z0/proto_z1: numpy arrays of same latent dimension or None
    - method: 'tsne' or 'umap' (umap requires umap-learn installed)
    """
    os.makedirs(out_dir, exist_ok=True)
    if latents is None or latents.shape[0] == 0:
        return

    N = latents.shape[0]
    labels_arr = np.array(labels)

    # sample if too many points
    if N > max_points:
        # sample balanced by label
        kept = []
        for lbl in np.unique(labels_arr):
            idxs = np.where(labels_arr == lbl)[0]
            k = max_points // max(1, len(np.unique(labels_arr)))
            if len(idxs) > k:
                chosen = np.random.choice(idxs, size=k, replace=False).tolist()
            else:
                chosen = idxs.tolist()
            kept.extend(chosen)
        kept = sorted(kept)
        latents_plot = latents[kept]
        labels_plot = labels_arr[kept]
    else:
        latents_plot = latents
        labels_plot = labels_arr

    # standardize
    try:
        scaler = StandardScaler()
        latents_plot = scaler.fit_transform(latents_plot)
    except Exception:
        pass

    # include prototypes in the embedding transform if present so they are mapped consistently
    proto_stack = []
    proto_labels = []

    # helper: coerce different prototype representations to 1D numpy arrays
    def _to_numpy_vec(v):
        if v is None:
            return None
        # torch tensor -> numpy
        try:
            import torch as _torch
        except Exception:
            _torch = None
        try:
            if _torch is not None and isinstance(v, _torch.Tensor):
                return v.detach().cpu().numpy().reshape(-1)
        except Exception:
            pass
        # numpy or list -> numpy 1D
        try:
            arr = np.array(v)
            return arr.reshape(-1)
        except Exception:
            return None

    p0 = _to_numpy_vec(proto_z0)
    p1 = _to_numpy_vec(proto_z1)

    if p0 is not None:
        proto_stack.append(np.atleast_2d(p0))
        proto_labels.append(-1)
    if p1 is not None:
        proto_stack.append(np.atleast_2d(p1))
        proto_labels.append(-2)

    try:
        if method == "umap" and _HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=random_state)
        else:
            reducer = TSNE(n_components=2, init='pca', random_state=random_state)

        if len(proto_stack) > 0:
            combined = np.vstack([latents_plot] + proto_stack)
            emb = reducer.fit_transform(combined)
            emb_points = emb[: latents_plot.shape[0]]
            emb_protos = emb[latents_plot.shape[0]:]
        else:
            emb_points = reducer.fit_transform(latents_plot)
            emb_protos = None
    except Exception as e:
        # fallback to PCA if t-SNE/umap fails
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            emb_points = pca.fit_transform(latents_plot)
            emb_protos = None
        except Exception:
            return

    fname_base = f"epoch{epoch}"
    if client_id is not None:
        fname_base += f"_client{client_id}"

    # write a small proto debug file so it's easy to see which prototypes were used (and their shapes)
    try:
        proto_info = {
            "proto_z0_present": p0 is not None,
            "proto_z1_present": p1 is not None,
            "proto_z0_shape": tuple(p0.shape) if p0 is not None else None,
            "proto_z1_shape": tuple(p1.shape) if p1 is not None else None,
            "emb_protos_count": int(emb_protos.shape[0]) if emb_protos is not None else 0,
        }

        # include raw prototype values (as lists) and distance between them if both present
        try:
            if p0 is not None:
                proto_info["proto_z0_values"] = p0.reshape(-1).astype(float).tolist()
            else:
                proto_info["proto_z0_values"] = None
            if p1 is not None:
                proto_info["proto_z1_values"] = p1.reshape(-1).astype(float).tolist()
            else:
                proto_info["proto_z1_values"] = None

            if p0 is not None and p1 is not None:
                try:
                    d = float(np.linalg.norm(p0 - p1))
                except Exception:
                    d = None
                proto_info["proto_z0z1_distance"] = d
                proto_info["proto_z0z1_same_eps_1e-6"] = (d is not None and d < 1e-6)
            else:
                proto_info["proto_z0z1_distance"] = None
                proto_info["proto_z0z1_same_eps_1e-6"] = None
        except Exception:
            # fallback if conversion fails
            proto_info["proto_z0_values"] = None
            proto_info["proto_z1_values"] = None
            proto_info["proto_z0z1_distance"] = None
            proto_info["proto_z0z1_same_eps_1e-6"] = None

        # include embedded prototype coordinates if available
        try:
            if emb_protos is not None:
                proto_info["emb_protos_coords"] = emb_protos.tolist()
            else:
                proto_info["emb_protos_coords"] = None
        except Exception:
            proto_info["emb_protos_coords"] = None

        import json as _json
        info_path = os.path.join(out_dir, f"{fname_base}_proto_info.json")
        with open(info_path, "w") as _jf:
            _json.dump(proto_info, _jf, indent=2)
    except Exception:
        pass

    plt.figure(figsize=(7, 6))
    unique_lbls = sorted(np.unique(labels_plot))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_lbls))))
    for i, lbl in enumerate(unique_lbls):
        mask = labels_plot == lbl
        plt.scatter(emb_points[mask, 0], emb_points[mask, 1], s=8, c=[colors[i]], label=str(lbl), alpha=0.7)

    # overlay prototypes
    if emb_protos is not None:
        for pidx, plabel in enumerate(proto_labels):
            px, py = emb_protos[pidx]
            if plabel == -1:
                plt.scatter(px, py, c='green', marker='X', s=120, edgecolor='k', label='proto_z0')
            elif plabel == -2:
                plt.scatter(px, py, c='red', marker='X', s=120, edgecolor='k', label='proto_z1')

    plt.legend(markerscale=2)
    plt.title(f"Latent embedding ({method}) - epoch {epoch}" + (f" client {client_id}" if client_id is not None else ""))
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{fname_base}_{method}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def plot_latent_embedding_non_iid_dir(latents, labels, seen_classes, attack_label=1,
                                      out_dir="visual", epoch=0, client_id=None,
                                      proto_z0=None, proto_z1=None, method="tsne",
                                      max_points=1000, random_state=42):
    """
        Variant of plot_latent_embedding for non-iid-dir experiments.
        Supports a single attack label (int) or multiple attack labels (list/tuple/set/ndarray).
        Colors points into three categories:
            - benign (label 0)
            - attack_seen: labels in intersection of attack_labels and seen_classes
            - attack_unseen: labels in attack_labels but NOT in seen_classes

        Parameters:
            - latents, labels: same as plot_latent_embedding
            - seen_classes: iterable of class ids that this client has seen (set/list)
            - attack_label: int or iterable of attack ids to split into seen/unseen
    """
    os.makedirs(out_dir, exist_ok=True)
    if latents is None or latents.shape[0] == 0:
        return

    labels_arr = np.array(labels)
    seen_set = set(int(x) for x in (seen_classes or []))

    # normalize attack_label to a set of ints (allow int or iterable)
    if isinstance(attack_label, (list, tuple, set, np.ndarray)):
        attack_labels_set = set(int(x) for x in attack_label)
    else:
        try:
            attack_labels_set = {int(attack_label)}
        except Exception:
            attack_labels_set = set()

    # Keep only samples that are either benign (0) or any of the chosen attack labels.
    keep_vals = [0] + sorted(attack_labels_set)
    keep_mask = np.isin(labels_arr, keep_vals)
    if not np.any(keep_mask):
        return None

    labels_arr = labels_arr[keep_mask]
    latents = latents[keep_mask]

    # Build 3-way category labels for plotting:
    # 0 = benign, 1 = attack_seen (label in attack_labels_set ∩ seen_set), 2 = attack_unseen (label in attack_labels_set \ seen_set)
    cat_labels = np.full(labels_arr.shape, 2, dtype=int)
    cat_labels[labels_arr == 0] = 0
    attack_idxs = np.where(np.isin(labels_arr, list(attack_labels_set)))[0]
    for i in attack_idxs:
        cat_labels[i] = 1 if int(labels_arr[i]) in seen_set else 2

    # limit number of points similar to plot_latent_embedding (balanced by original label groups)
    N = latents.shape[0]
    if N > max_points:
        kept = []
        unique_orig = np.unique(labels_arr)
        k_per = max_points // max(1, len(unique_orig))
        for lbl in unique_orig:
            idxs = np.where(labels_arr == lbl)[0]
            if len(idxs) > k_per:
                chosen = np.random.choice(idxs, size=k_per, replace=False).tolist()
            else:
                chosen = idxs.tolist()
            kept.extend(chosen)
        kept = sorted(kept)
        latents_plot = latents[kept]
        labels_plot = labels_arr[kept]
        cat_plot = cat_labels[kept]
    else:
        latents_plot = latents
        labels_plot = labels_arr
        cat_plot = cat_labels

    # standardize
    try:
        scaler = StandardScaler()
        latents_plot = scaler.fit_transform(latents_plot)
    except Exception:
        pass

    # helper to coerce prototypes
    def _to_numpy_vec(v):
        if v is None:
            return None
        try:
            import torch as _torch
        except Exception:
            _torch = None
        try:
            if _torch is not None and isinstance(v, _torch.Tensor):
                return v.detach().cpu().numpy().reshape(-1)
        except Exception:
            pass
        try:
            arr = np.array(v)
            return arr.reshape(-1)
        except Exception:
            return None

    p0 = _to_numpy_vec(proto_z0)
    p1 = _to_numpy_vec(proto_z1)
    proto_stack = []
    proto_labels = []
    if p0 is not None:
        proto_stack.append(np.atleast_2d(p0)); proto_labels.append(-1)
    if p1 is not None:
        proto_stack.append(np.atleast_2d(p1)); proto_labels.append(-2)

    try:
        if method == "umap" and _HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=random_state)
        else:
            reducer = TSNE(n_components=2, init='pca', random_state=random_state)

        if len(proto_stack) > 0:
            combined = np.vstack([latents_plot] + proto_stack)
            emb = reducer.fit_transform(combined)
            emb_points = emb[: latents_plot.shape[0]]
            emb_protos = emb[latents_plot.shape[0]:]
        else:
            emb_points = reducer.fit_transform(latents_plot)
            emb_protos = None
    except Exception:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            emb_points = pca.fit_transform(latents_plot)
            emb_protos = None
        except Exception:
            return

    fname_base = f"epoch{epoch}"
    if client_id is not None:
        fname_base += f"_client{client_id}"

    # write proto info
    try:
        proto_info = {"proto_z0_present": p0 is not None, "proto_z1_present": p1 is not None,
                      "seen_classes": list(sorted(seen_set)),
                      "attack_labels": list(sorted(attack_labels_set))}
        import json as _json
        info_path = os.path.join(out_dir, f"{fname_base}_non_iid_proto_info.json")
        with open(info_path, "w") as _jf:
            _json.dump(proto_info, _jf, indent=2)
    except Exception:
        pass

    # Plot: three categories only (benign, attack_seen, attack_unseen)
    plt.figure(figsize=(7, 6))
    palette = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:red'}
    # summarize attack labels for legend clarity
    try:
        atk_str = ','.join(str(a) for a in sorted(attack_labels_set)) if len(attack_labels_set) > 0 else 'N/A'
    except Exception:
        atk_str = 'N/A'
    labels_for_legend = {0: 'benign', 1: f'attack_seen ({atk_str})', 2: f'attack_unseen ({atk_str})'}
    unique_cats = sorted(np.unique(cat_plot))
    for c in unique_cats:
        mask = cat_plot == c
        plt.scatter(emb_points[mask, 0], emb_points[mask, 1], s=8, c=palette.get(c, 'black'), label=labels_for_legend.get(c, str(c)), alpha=0.8)

    # overlay prototypes
    if emb_protos is not None:
        for pidx, plabel in enumerate(proto_labels):
            px, py = emb_protos[pidx]
            if plabel == -1:
                plt.scatter(px, py, c='green', marker='X', s=120, edgecolor='k', label='proto_z0')
            elif plabel == -2:
                plt.scatter(px, py, c='red', marker='X', s=120, edgecolor='k', label='proto_z1')

    plt.legend(markerscale=2)
    plt.title(f"Latent embedding (non-iid-dir) ({method}) - epoch {epoch}" + (f" client {client_id}" if client_id is not None else ""))
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{fname_base}_{method}_non_iid.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def plot_latent_first_component_hist_from_latents(latents, out_dir, epoch, client_id=None, bins=50):
    """
    Plot and save a histogram of the first component (index 0) of the latent vectors.
    Also save the raw values as a CSV for further analysis.

    - latents: numpy array shape (N, D)
    - out_dir: directory to save outputs
    - epoch, client_id: used for filenames
    - bins: number of histogram bins
    Returns path to saved PNG or None if nothing saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    if latents is None or latents.shape[0] == 0:
        return None

    try:
        first_comp = np.asarray(latents)[:, 0]
    except Exception:
        return None

    fname_base = f"epoch{epoch}"
    if client_id is not None:
        fname_base += f"_client{client_id}"

    # histogram
    try:
        plt.figure(figsize=(6, 4))
        plt.hist(first_comp, bins=bins, color="C2", alpha=0.85)
        plt.title(f"Latent dim[0] histogram - epoch {epoch}" + (f" client {client_id}" if client_id is not None else ""))
        plt.xlabel("latent[0]")
        plt.ylabel("count")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{fname_base}_latent_dim0_hist.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return None


def plot_latent_first_component_hist_from_client(client, out_dir, epoch, client_id=None, max_samples_per_class=None, bins=50):
    """
    Convenience wrapper: collect latents from a client and plot the first-dimension histogram.
    """
    try:
        latents, _ = collect_latents_and_labels_from_client(client, max_samples_per_class=max_samples_per_class)
        return plot_latent_first_component_hist_from_latents(latents, out_dir, epoch, client_id=client_id, bins=bins)
    except Exception:
        return None
