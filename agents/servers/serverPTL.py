import os
import copy
import torch
import numpy as np
from collections import defaultdict

from function.utils import (
    average_nn_parameters,
    attention_average_nn_parameters,
    kmeans_cluster_parameters,
    kmeans_cluster_parameters_and_get_min_center,
)
from function.utils.visualize_util import (
    log_re_distributions_from_lists,
    collect_re_and_labels_from_client,
    collect_z_norms_and_labels_from_client,
    log_z_boxplots_from_lists,
    collect_latents_and_labels_from_client,
    plot_latent_embedding,
    plot_latent_embedding_non_iid_dir,
    plot_latent_first_component_hist_from_latents,
    plot_latent_first_component_hist_from_client,
    compute_auc_per_attack_from_flat,
)
import json


class ServerPTL:
    def __init__(self, args):
        self.args = args

        # prototype vectors (maintained on server as numpy arrays)
        self.proto_z0 = None
        self.proto_z1 = None

        # ema factor for prototype updates
        self.ema = getattr(args, "ptl_proto_ema", 0.9)

    def train_on_clients(self, epoch, clients, poisoned_workers):
        self.args.logger.info("Training {} model epoch #{}", self.args.model_type, str(epoch))

        list_loss = []
        list_client_training = []
        proto_stats_list = []

        for client_idx, client in enumerate(clients):
            if client.is_training:
                list_client_training.append(client_idx)

                # client.train returns (avg_loss, local_proto_stats)
                result = client.train(epoch)
                if isinstance(result, tuple):
                    train_loss, local_proto_stats = result
                else:
                    train_loss = result
                    local_proto_stats = None

                list_loss.append(train_loss if not hasattr(train_loss, 'item') else train_loss.item())

                val_loss, threshold_re, threshold_z = client.validate(epoch)
                client.set_recent_metric(getattr(client, 'recent_re', 0.0), None, train_loss if not hasattr(train_loss, 'item') else train_loss.item(), val_loss, threshold_re, threshold_z)

                if val_loss < client.best_loss * 1.01:
                    best_weight_model = copy.deepcopy(client.get_nn_parameters())
                    client.set_best_ckpt(val_loss, epoch, threshold_re, threshold_z, best_weight_model)
                    self.args.logger.info(
                        "Client {} gets new best val_loss at epoch #{}: {:.6f}",
                        str(client_idx), str(client.best_epoch), client.best_loss,
                    )
                elif client.best_epoch + self.args.es_offset <= epoch:
                    client.set_training_status(False)
                    self.args.logger.info("Client {} early stopped at epoch #{}", str(client_idx), str(epoch))

                # save model periodically
                if client.is_training and epoch % 10 == 0:
                    save_dir = f"saved_models/{self.args.model_type}/{self.args.num_multi_class_clients}/{self.args.aggregation_type}/{self.args.dataset}/"
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(save_dir, f"epoch_{epoch}_client_{client_idx}.pt")
                    torch.save(client.net.state_dict(), model_path)
                    self.args.logger.info(f"Saved model for client {client_idx} at epoch {epoch} -> {model_path}")

                proto_stats_list.append(local_proto_stats)

                # log train progress
                self.log_train_progress(epoch, client_idx, client, client_idx in poisoned_workers)

        # If no clients were training, stop
        if len(list_client_training) == 0:
            return True

        # Aggregate prototypes from clients and update server prototypes
        self.receive_proto_stats_and_update(proto_stats_list)

        # Broadcast updated prototypes to all clients
        for c in clients:
            c.set_prototypes(self.proto_z0, self.proto_z1)

        # Aggregate model parameters from clients that trained
        parameters = [clients[i].get_nn_parameters() for i in list_client_training]

        if self.args.aggregation_type == "average":
            new_params = average_nn_parameters(parameters)
        elif self.args.aggregation_type == "attention":
            np_list_loss = np.asarray(list_loss, dtype=np.float32)
            loss_sum = np.sum(np_list_loss)
            list_weight_loss = np.log(loss_sum / np_list_loss)
            weight_loss_sum = np.sum(list_weight_loss)
            list_aggregation_coef = list_weight_loss / weight_loss_sum
            if len(parameters) == 1:
                list_aggregation_coef = [1.0]
            new_params = attention_average_nn_parameters(parameters, list_aggregation_coef)
        elif self.args.aggregation_type == "kmean":
            clustered = kmeans_cluster_parameters_and_get_min_center(parameters, list_loss, 2)
            new_params = average_nn_parameters(clustered)
        else:
            new_params = average_nn_parameters(parameters)

        # Update clients with new global params

        for client_idx in list_client_training:
            clients[client_idx].update_nn_parameters(new_params)

        # Train-time latent[0] histograms every 100 epochs for non_iid_dir, using TEST latents
        if getattr(self.args, 'experiment_type', None) == 'non_iid_dir' and (epoch < 10 or (epoch % 10 == 0 and epoch<100) or epoch % 100 == 0):
            out_dir = os.path.join(
                "logs",
                "re_distributions",
                f"{self.args.model_type}_mc{self.args.num_multi_class_clients}",
                "train_hist",
            )
            os.makedirs(out_dir, exist_ok=True)
            global_latent_firstcomp_train = []
            for client_idx, client in enumerate(clients):
                # collect full TEST latents for consistent distribution view
                latents, _ = collect_latents_and_labels_from_client(client, max_samples_per_class=None)
                if latents is not None and latents.shape[0] > 0:
                    client_out_dir = os.path.join(out_dir, f"client_{client_idx}")
                    os.makedirs(client_out_dir, exist_ok=True)
                    hist_p = plot_latent_first_component_hist_from_latents(latents, client_out_dir, epoch, client_id=client_idx)
                    if hist_p:
                        self.args.logger.info(f"[Train] Saved client {client_idx} latent-dim0 histogram to {hist_p}")
                    # accumulate for global train histogram
                    first_comp = np.asarray(latents)[:, 0].tolist()
                    global_latent_firstcomp_train.extend(first_comp)

            # Global train histogram
            if len(global_latent_firstcomp_train) > 0:
                arr = np.array(global_latent_firstcomp_train).reshape(-1, 1)
                global_hist_path = plot_latent_first_component_hist_from_latents(arr, out_dir, epoch, client_id=None)
                if global_hist_path:
                    self.args.logger.info(f"[Train] Saved global latent-dim0 histogram to {global_hist_path}")

        return False

    def receive_proto_stats_and_update(self, proto_stats_list):
        # proto_stats_list: list of dicts from clients: {label: (sum_vec_numpy, count)}
        # aggregate sums and counts
        sums = defaultdict(lambda: None)
        counts = defaultdict(int)
        for stats in proto_stats_list:
            if stats is None:
                continue
            for label, (sum_vec, cnt) in stats.items():
                if sums[label] is None:
                    sums[label] = np.array(sum_vec, dtype=float)
                else:
                    sums[label] += np.array(sum_vec, dtype=float)
                counts[label] += int(cnt)

        # compute mean prototypes for labels present (0 and 1 expected)
        proto_means = {}
        for label, s in sums.items():
            proto_means[label] = s / max(1, counts[label])

        # update server prototypes via EMA
        if 0 in proto_means:
            z0_new = proto_means[0]
            if self.proto_z0 is None:
                self.proto_z0 = z0_new
            else:
                self.proto_z0 = self.ema * self.proto_z0 + (1 - self.ema) * z0_new

        if 1 in proto_means:
            z1_new = proto_means[1]
            if self.proto_z1 is None:
                self.proto_z1 = z1_new
            else:
                self.proto_z1 = self.ema * self.proto_z1 + (1 - self.ema) * z1_new

    def test_on_clients(self, epoch, clients, poisoned_workers):
        self.args.logger.info("Testing {} model at epoch #{}", self.args.model_type, str(epoch))

        multipliers = np.arange(0.0, 5.1, 0.2)

        multiplier_auc_all = {m: [] for m in multipliers}
        multiplier_auc_benign = {m: [] for m in multipliers}
        multiplier_auc_poisoned = {m: [] for m in multipliers}
        # accumulate global RE and labels across clients for epoch-level analysis
        global_re = []
        global_labels = []
        # keep track of which client each sample came from (needed for seen/unseen grouping)
        global_client_idxs = []
        # collect first-dimension values across clients for global histogram in non_iid_dir
        global_latent_firstcomp = []
        # also store per-client raw data to compute per-client seen/unseen metrics later
        per_client_data = {}

        for client_idx, client in enumerate(clients):
            self.args.logger.info(
                "Client {} test params: threshold_re (mean={:.6f}, std={:.6f}), best epoch {}".format(
                    client_idx, client.threshold_re[0], client.threshold_re[1], client.best_epoch
                )
            )

            recent_weight_model = client.get_nn_parameters()
            client.update_nn_parameters(client.best_weight_model)

            # Ensure client uses the prototypes corresponding to its best checkpoint (if available)
            bp0 = getattr(client, 'best_proto_z0', None)
            bp1 = getattr(client, 'best_proto_z1', None)
            client.set_prototypes(bp0, bp1)

            # Collect per-sample RE and raw labels from client's test set
            client_re_list, client_raw_labels = collect_re_and_labels_from_client(client)

            # Build out_dir name using model and multi-class setting
            out_dir = os.path.join(
                "logs",
                "re_distributions",
                f"{self.args.model_type}_mc{self.args.num_multi_class_clients}_epoch_{epoch}",
            )
            client_out_dir = os.path.join(out_dir, f"client_{client_idx}")

            # Save per-client RE distributions (CSV + plots)
            log_re_distributions_from_lists(client_re_list, client_raw_labels, client_out_dir, epoch, client_id=client_idx)

            # Also collect & log latent (z) boxplots (L2 norms) per-client
            client_z_list, client_z_labels = collect_z_norms_and_labels_from_client(client)
            log_z_boxplots_from_lists(client_z_list, client_z_labels, client_out_dir, epoch, client_id=client_idx)

            # Collect full latent vectors and produce t-SNE/UMAP plots with prototype overlay
            latents, lat_labels = collect_latents_and_labels_from_client(client, max_samples_per_class=1000)
            if latents is not None and latents.shape[0] > 0:
                plot_latent_embedding(latents, lat_labels, client_out_dir, epoch, client_id=client_idx, proto_z0=bp0, proto_z1=bp1, method='tsne')
                # For non-iid-dir experiments, also produce the specialized 3-color embedding
                if getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
                    # obtain seen_set for this client from partition metadata if available
                    # Always use TRAIN partition metadata for seen/unseen determination
                    pm_train = getattr(self.args, 'train_partition_meta', None) or {}
                    seen_list = pm_train.get('seen_sets', []) if isinstance(pm_train, dict) else []
                    seen_for_client = []
                    if isinstance(seen_list, list) and client_idx < len(seen_list):
                        seen_for_client = list(map(int, seen_list[client_idx]))

                    # build full set of attack labels from args if available; fallback to unique labels
                    num_attacks = getattr(self.args, 'num_attack_labels', None)
                    if isinstance(num_attacks, int) and num_attacks > 0:
                        attack_labels = list(range(1, num_attacks + 1))
                    else:
                        # fallback: infer from labels present
                        uniq = sorted(set(int(x) for x in lat_labels if int(x) != 0))
                        attack_labels = uniq

                    out = plot_latent_embedding_non_iid_dir(latents, lat_labels, seen_for_client,
                                                            attack_label=attack_labels,
                                                            out_dir=client_out_dir,
                                                            epoch=epoch,
                                                            client_id=client_idx,
                                                            proto_z0=bp0,
                                                            proto_z1=bp1,
                                                            method='tsne',
                                                            max_points=1000,
                                                            random_state=getattr(self.args, 'assign_seed', 0))
                    if out:
                        self.args.logger.info(f"Saved non-iid-dir latent viz for client {client_idx} -> {out}")
                # if non-iid-dir experiments, save histogram of first latent component per-client
                if getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
                    # per-client histogram
                    hist_p = plot_latent_first_component_hist_from_latents(latents, client_out_dir, epoch, client_id=client_idx)
                    if hist_p:
                        self.args.logger.info(f"Saved client {client_idx} latent-dim0 histogram to {hist_p}")
                    # accumulate for global histogram
                    first_comp = np.asarray(latents)[:, 0].tolist()
                    global_latent_firstcomp.extend(first_comp)

            # accumulate for global aggregation and keep per-client data
            global_re.extend(client_re_list)
            global_labels.extend(client_raw_labels)
            global_client_idxs.extend([client_idx] * len(client_raw_labels))
            per_client_data[client_idx] = {
                "re": list(map(float, client_re_list)),
                "labels": list(map(int, client_raw_labels)),
                "threshold_re": float(client.threshold_re[0]) if isinstance(client.threshold_re, (list, tuple)) else float(client.threshold_re),
            }

            acc_list, precision_list, recall_list, f1_list, auc_list = client.test()

            client.update_nn_parameters(recent_weight_model)

            for i, m in enumerate(multipliers):
                acc, precision, recall, f1, auc = acc_list[i], precision_list[i], recall_list[i], f1_list[i], auc_list[i]

                if m == 0.0:
                    self.args.logger.info(
                        f"[Client {client_idx}] Multiplier {m:.1f}: ACC={acc:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
                    )

                # store per-threshold results
                import pandas as pd
                self.args.set_test_log_df(
                    pd.concat(
                        [
                            self.args.get_test_log_df(),
                            pd.DataFrame(
                                [
                                    {
                                        "epoch": epoch,
                                        "client_id": client_idx,
                                        "is_mal": client_idx in poisoned_workers,
                                        "threshold_multiplier": round(m, 1),
                                        "auc": auc,
                                        "accuracy": acc,
                                        "precision": precision,
                                        "recall": recall,
                                        "f1": f1,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                )

                multiplier_auc_all[m].append(auc)
                if client_idx in poisoned_workers:
                    multiplier_auc_poisoned[m].append(auc)
                else:
                    multiplier_auc_benign[m].append(auc)

        # After per-client collection, produce global RE plots and per-attack AUCs
        if 'global_re' in locals() and len(global_re) > 0:
            out_dir = os.path.join(
                "logs",
                "re_distributions",
                f"{self.args.model_type}_mc{self.args.num_multi_class_clients}_epoch_{epoch}",
            )
            summary, grouped = log_re_distributions_from_lists(global_re, global_labels, out_dir, epoch, client_id=None)

            # global latent z norms + boxplots
            global_z_list = []
            # collect latent z norms across clients
            for client in clients:
                z_list, z_labels = collect_z_norms_and_labels_from_client(client)
                global_z_list.extend(z_list)
            if len(global_z_list) > 0:
                log_z_boxplots_from_lists(global_z_list, global_labels, out_dir, epoch, client_id=None)

            # global histogram of latent first component (for non_iid_dir experiments)
            if getattr(self.args, 'experiment_type', None) == 'non_iid_dir' and len(global_latent_firstcomp) > 0:
                arr = np.array(global_latent_firstcomp).reshape(-1, 1)
                global_hist_path = plot_latent_first_component_hist_from_latents(arr, out_dir, epoch, client_id=None)
                if global_hist_path:
                    self.args.logger.info(f"Saved global latent-dim0 histogram to {global_hist_path}")

            aucs = compute_auc_per_attack_from_flat(global_labels, global_re)
            auc_path = os.path.join(out_dir, f"epoch{epoch}_aucs_per_attack.json")
            # convert keys to strings for JSON stability
            aucs_json = {str(int(k)): (None if v is None else float(v)) for k, v in aucs.items()}
            with open(auc_path, "w") as jf:
                json.dump(aucs_json, jf, indent=2)
            self.args.logger.info(f"Saved per-attack AUCs to {auc_path}")

            # If running non_iid_dir experiments, compute per-client and global seen/unseen metrics
            if getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
                    # load seen_sets from partition metadata if available
                    pm_train = getattr(self.args, 'train_partition_meta', None) or {}
                    seen_sets = pm_train.get('seen_sets', [])

                    # per-client: compute per-attack AUCs and classification metrics (using client threshold)
                    for cidx, pdata in per_client_data.items():
                        client_dir = os.path.join(out_dir, f"client_{cidx}")
                        os.makedirs(client_dir, exist_ok=True)
                        # per-attack AUCs for this client
                        client_aucs = compute_auc_per_attack_from_flat(pdata['labels'], pdata['re'])
                        client_aucs_json = {str(int(k)): (None if v is None else float(v)) for k, v in client_aucs.items()}

                        # classification metrics per attack using client's threshold
                        client_obj = clients[cidx]
                        classif = client_obj.test_by_attack_type_full(pdata['threshold_re'], None, verbose=False)

                        # attach seen set if exists (map by multi-class client index)
                        seen_set = []
                        if isinstance(seen_sets, list) and cidx < len(seen_sets):
                            seen_set = list(map(int, seen_sets[cidx]))

                        # Write per-attack metrics to CSV for easier comparison
                        import pandas as pd
                        rows = []
                        # merge AUC and classification metrics per attack
                        for atk_str, auc_val in client_aucs_json.items():
                            atk_id = int(atk_str)
                            cls_metrics = classif.get(atk_id, {})
                            rows.append({
                                'epoch': int(epoch),
                                'client_id': int(cidx),
                                'attack_type': atk_id,
                                'auc': auc_val if auc_val is not None else 0.0,
                                'accuracy': float(cls_metrics.get('acc', 0.0)),
                                'precision': float(cls_metrics.get('precision', 0.0)),
                                'recall': float(cls_metrics.get('recall', 0.0)),
                                'f1': float(cls_metrics.get('f1-score', 0.0)),
                                'support': int(cls_metrics.get('support', 0)),
                            })

                        client_metrics_csv = os.path.join(client_dir, f"epoch{epoch}_client{cidx}_per_attack_metrics.csv")
                        pd.DataFrame(rows).to_csv(client_metrics_csv, index=False)
                        self.args.logger.info(f"Saved per-attack CSV metrics for client {cidx} -> {client_metrics_csv}")

                    # global seen/unseen grouping for a chosen attack label (default 1)
                    attack_label = int(getattr(self.args, 'attack_label', 1))
                    # compute a global threshold (mean of client thresholds) for classification f1/precision/recall
                    global_threshold = float(np.mean([pdata.get('threshold_re', 0.0) for pdata in per_client_data.values()]))

                    # build per-sample seen flag
                    seen_flag = []
                    for ci in global_client_idxs:
                        if isinstance(seen_sets, list) and ci < len(seen_sets):
                            seen_flag.append(int(attack_label in list(map(int, seen_sets[ci]))))
                        else:
                            seen_flag.append(0)

                    # compute seen/unseen metrics: for attack_label compare against benign (0)
                    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

                    def _compute_group_metrics(mask_idxs):
                        if not any(mask_idxs):
                            return {'auc': None, 'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'support_attack': 0, 'support_benign': 0}
                        idxs = np.where(np.array(mask_idxs))[0]
                        y = np.array(global_labels)[idxs]
                        scores = np.array(global_re)[idxs]
                        # restrict to labels 0 or attack_label
                        keep_mask = np.isin(y, [0, attack_label])
                        if keep_mask.sum() == 0:
                            return {'auc': None, 'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'support_attack': 0, 'support_benign': 0}
                        y2 = y[keep_mask]
                        scores2 = scores[keep_mask]
                        y_bin = (y2 == attack_label).astype(int)
                        # AUC
                        auc_v = float(roc_auc_score(y_bin, scores2)) if len(np.unique(y_bin)) > 1 else None
                        # classification at global_threshold
                        preds = (scores2 > global_threshold).astype(int)
                        acc_v = float(accuracy_score(y_bin, preds))
                        prec_v = float(precision_score(y_bin, preds, zero_division=0))
                        rec_v = float(recall_score(y_bin, preds, zero_division=0))
                        f1_v = float(f1_score(y_bin, preds, zero_division=0))
                        return {'auc': auc_v, 'accuracy': acc_v, 'precision': prec_v, 'recall': rec_v, 'f1': f1_v, 'support_attack': int((y2 == attack_label).sum()), 'support_benign': int((y2 == 0).sum())}

                    # masks for attack_seen and attack_unseen plus benign (we'll construct masks where sample belongs to either benign or the relevant attack subset)
                    labels_arr = np.array(global_labels)
                    seen_flags_arr = np.array(seen_flag)

                    # mask for samples that are either benign or attack_label
                    mask_attack_samples = np.isin(labels_arr, [0, attack_label])

                    # build masks for seen-group and unseen-group (True where sample included)
                    seen_group_mask = mask_attack_samples & ((labels_arr == 0) | ((labels_arr == attack_label) & (seen_flags_arr == 1)))
                    unseen_group_mask = mask_attack_samples & ((labels_arr == 0) | ((labels_arr == attack_label) & (seen_flags_arr == 0)))

                    seen_metrics = _compute_group_metrics(seen_group_mask)
                    unseen_metrics = _compute_group_metrics(unseen_group_mask)

                    global_seen_payload = {
                        'attack_label': int(attack_label),
                        'global_threshold_used': float(global_threshold),
                        'seen_group': seen_metrics,
                        'unseen_group': unseen_metrics,
                    }
                    seen_path = os.path.join(out_dir, f"epoch{epoch}_seen_unseen_global_attack{attack_label}.json")
                    with open(seen_path, 'w') as _jf:
                        json.dump(global_seen_payload, _jf, indent=2)
                    self.args.logger.info(f"Saved global seen/unseen metrics to {seen_path}")

        self._log_avg_auc(multiplier_auc_all, multiplier_auc_benign, multiplier_auc_poisoned)

    def _log_avg_auc(self, multiplier_auc_all, multiplier_auc_benign, multiplier_auc_poisoned):
        header = "\n====== AVERAGE AUC PER MULTIPLIER ======\n"
        header += "{:<12} {:<20} {:<20} {:<20}\n".format("Multiplier", "All Clients", "Benign Clients", "Poisoned Clients")
        header += "-" * 75 + "\n"

        rows = ""
        for m in sorted(multiplier_auc_all.keys()):
            all_avg = np.mean(multiplier_auc_all[m]) if multiplier_auc_all[m] else 0
            benign_avg = np.mean(multiplier_auc_benign[m]) if multiplier_auc_benign[m] else 0
            poisoned_avg = np.mean(multiplier_auc_poisoned[m]) if multiplier_auc_poisoned[m] else 0

            rows += "{:<12.1f} {:<20.6f} {:<20.6f} {:<20.6f}\n".format(
                m, all_avg, benign_avg, poisoned_avg
            )

        self.args.logger.info(header + rows)

    def log_train_progress(self, epoch, client_idx, client, is_mal):
        import pandas as pd
        # prepare per-client recent prototypes for logging (convert to python lists or None)
        try:
            rp0 = getattr(client, 'prototype_z0', None)
            rp1 = getattr(client, 'prototype_z1', None)
            recent_proto_z0 = rp0.tolist() if (rp0 is not None and hasattr(rp0, 'tolist')) else (list(rp0) if rp0 is not None else None)
            recent_proto_z1 = rp1.tolist() if (rp1 is not None and hasattr(rp1, 'tolist')) else (list(rp1) if rp1 is not None else None)
        except Exception:
            recent_proto_z0, recent_proto_z1 = None, None

        # prepare best prototypes for logging
        try:
            bp0 = getattr(client, 'best_proto_z0', None)
            bp1 = getattr(client, 'best_proto_z1', None)
            best_proto_z0 = bp0.tolist() if (bp0 is not None and hasattr(bp0, 'tolist')) else (list(bp0) if bp0 is not None else None)
            best_proto_z1 = bp1.tolist() if (bp1 is not None and hasattr(bp1, 'tolist')) else (list(bp1) if bp1 is not None else None)
        except Exception:
            best_proto_z0, best_proto_z1 = None, None
        new_row = pd.DataFrame([
            {
                "epoch": epoch,
                "client_id": client_idx,
                "is_mal": is_mal,
                "train_re": getattr(client, 'recent_re', 0.0),
                "train_latent_z": getattr(client, 'recent_latent_z', 0.0),
                "train_loss": getattr(client, 'recent_train_loss', 0.0),
                "val_loss": getattr(client, 'recent_val_loss', 0.0),
                "threshold_re": getattr(client, 'recent_threshold_re', (0,0)),
                "proto_z0": recent_proto_z0,
                "proto_z1": recent_proto_z1,
                "best_proto_z0": best_proto_z0,
                "best_proto_z1": best_proto_z1,
                "best_val_loss": client.best_loss if hasattr(client, 'best_loss') else 0,
                "best_epoch": client.best_epoch if hasattr(client, 'best_epoch') else -1,
                "is_training": client.is_training,
            }
        ])
        self.args.set_train_log_df(pd.concat([self.args.get_train_log_df(), new_row], ignore_index=True))
