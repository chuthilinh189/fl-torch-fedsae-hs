import os
import copy
import math
import numpy as np
import pandas as pd
import torch
import json
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
    collect_predictions_and_labels_from_client,
    plot_latent_embedding,
    plot_latent_embedding_non_iid_dir,
    plot_latent_first_component_hist_from_latents,
    compute_auc_per_attack_from_flat,
)


class ServerFedHome:
    def __init__(self, args):
        self.args = args

    def train_on_clients(self, epoch, clients, poisoned_workers):
        random_workers = list(range(self.args.num_workers))
        self.args.logger.info("Training {} model epoch #{}", self.args.model_type, str(epoch))

        list_loss = []
        list_client_training = []

        for client_idx in random_workers:
            client = clients[client_idx]
            if client.is_training:
                list_client_training.append(client_idx)

                total_loss = client.train(epoch)
                list_loss.append(float(total_loss.item()))

                val_loss = client.validate(epoch)
                client.set_recent_metric(client.recent_re, client.recent_ce, None, float(total_loss.item()), val_loss)

                if val_loss < client.best_loss:
                    best_weight_model = copy.deepcopy(client.get_nn_parameters())
                    client.set_best_ckpt(val_loss, epoch, best_weight_model)
                    self.args.logger.info(
                        "Client {} gets new best val_loss at epoch #{}: {:.6f}",
                        str(client_idx), str(client.best_epoch), client.best_loss,
                    )
                elif client.best_epoch + self.args.es_offset <= epoch:
                    client.set_training_status(False)
                    self.args.logger.info("Client {} early stopped at epoch #{}", str(client_idx), str(epoch))

                if client.is_training and epoch % 10 == 0:
                    save_dir = f"saved_models/{self.args.model_type}/{self.args.num_multi_class_clients}/{self.args.aggregation_type}/{self.args.dataset}/"
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(save_dir, f"epoch_{epoch}_client_{client_idx}.pt")
                    torch.save(client.net.state_dict(), model_path)
                    self.args.logger.info(f"Saved model for client {client_idx} at epoch {epoch} -> {model_path}")

                self.log_train_progress(epoch, client_idx, client, client_idx in poisoned_workers)

        self.args.logger.info("{} clients still training at epoch #{}", str(len(list_client_training)), str(epoch))

        if len(list_client_training) > 0:
            parameters = [clients[cid].get_nn_parameters() for cid in list_client_training]

            if self.args.aggregation_type == "cluster" and len(list_client_training) > 1:
                cluster_result = kmeans_cluster_parameters(parameters, n_clusters=2)
                averaged_params = cluster_result["clustered_params"]
                cluster_assignments = cluster_result["cluster_assignments"]

                cluster_0 = [list_client_training[i] for i in range(len(cluster_assignments)) if cluster_assignments[i] == 0]
                cluster_1 = [list_client_training[i] for i in range(len(cluster_assignments)) if cluster_assignments[i] == 1]
                self.args.logger.info(f"Cluster 0 clients: {cluster_0}")
                self.args.logger.info(f"Cluster 1 clients: {cluster_1}")

                for idx, client_idx in enumerate(list_client_training):
                    if cluster_assignments[idx] == 0:
                        clients[client_idx].update_nn_parameters(averaged_params[0])
                    else:
                        clients[client_idx].update_nn_parameters(averaged_params[1])
            else:
                new_nn_params = self.aggregate_parameters(parameters, list_loss)
            for client_idx in list_client_training:
                clients[client_idx].update_nn_parameters(new_nn_params)

        return len(list_client_training) == 0

    def aggregate_parameters(self, parameters, list_loss):
        if self.args.aggregation_type == "average":
            return average_nn_parameters(parameters)
        if self.args.aggregation_type == "attention":
            np_list_loss = np.asarray(list_loss, dtype=np.float32)
            loss_sum = np.sum(np_list_loss)
            list_weight_loss = np.log(loss_sum / np_list_loss)
            weight_loss_sum = np.sum(list_weight_loss)
            list_aggregation_coef = list_weight_loss / weight_loss_sum
            if len(parameters) == 1:
                list_aggregation_coef = [1.0]
            return attention_average_nn_parameters(parameters, list_aggregation_coef)
        if self.args.aggregation_type == "split":
            k = 0.2
            num_keep = math.ceil(len(list_loss) * (1 - k))
            sorted_losses_params = sorted(zip(list_loss, parameters), key=lambda x: x[0])
            selected_params = [p for l, p in sorted_losses_params[:num_keep]]
            return average_nn_parameters(selected_params)
        if self.args.aggregation_type == "kmean":
            clustered_params = kmeans_cluster_parameters_and_get_min_center(parameters, list_loss, 2)
            return average_nn_parameters(clustered_params)
        raise ValueError(f"Unsupported aggregation type: {self.args.aggregation_type}")

    def test_on_clients(self, epoch, clients, poisoned_workers):
        self.args.logger.info("Testing {} model at epoch #{}", self.args.model_type, str(epoch))

        multipliers = [0.0]
        multiplier_auc_all = {m: [] for m in multipliers}
        multiplier_auc_benign = {m: [] for m in multipliers}
        multiplier_auc_poisoned = {m: [] for m in multipliers}

        # non-iid-dir support
        global_re = []
        global_labels = []
        global_client_idxs = []
        global_latent_firstcomp = []
        per_client_data = {}
        detailed_metrics_rows = []

        for client_idx, client in enumerate(clients):
            self.args.logger.info("Client {} test params: best epoch {}".format(client_idx, client.best_epoch))

            recent_weight_model = client.get_nn_parameters()
            client.update_nn_parameters(client.best_weight_model)

            acc_list, precision_list, recall_list, f1_list, auc_list = client.test()
            acc = acc_list[0]
            precision = precision_list[0]
            recall = recall_list[0]
            f1 = f1_list[0]
            auc = auc_list[0]

            client.update_nn_parameters(recent_weight_model)

            self.args.logger.info(
                f"[Client {client_idx}] ACC={acc:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
            )

            existing_df = self.args.get_test_log_df()
            new_row = pd.DataFrame([
                {
                    "epoch": epoch,
                    "client_id": client_idx,
                    "is_mal": client_idx in poisoned_workers,
                    "threshold_multiplier": multipliers[0],
                    "auc": auc,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            ])
            if existing_df is None or (isinstance(existing_df, pd.DataFrame) and existing_df.empty):
                self.args.set_test_log_df(new_row)
            else:
                self.args.set_test_log_df(pd.concat([existing_df, new_row], ignore_index=True))

            multiplier_auc_all[multipliers[0]].append(auc)
            if client_idx in poisoned_workers:
                multiplier_auc_poisoned[multipliers[0]].append(auc)
            else:
                multiplier_auc_benign[multipliers[0]].append(auc)

            out_dir = os.path.join(
                "logs",
                "re_distributions",
                f"{self.args.model_type}_mc{self.args.num_multi_class_clients}_epoch_{epoch}",
            )
            client_dir = os.path.join(out_dir, f"client_{client_idx}")
            os.makedirs(client_dir, exist_ok=True)
            classif = client.test_by_attack_type_full(verbose=False)
            rows = []
            for atk_id, metrics in classif.items():
                rows.append(
                    {
                        "epoch": int(epoch),
                        "client_id": int(client_idx),
                        "attack_type": int(atk_id),
                        "auc": float(auc),
                        "accuracy": float(metrics.get("acc", 0.0)),
                        "precision": float(metrics.get("precision", 0.0)),
                        "recall": float(metrics.get("recall", 0.0)),
                        "f1": float(metrics.get("f1-score", 0.0)),
                        "support": int(metrics.get("support", 0)),
                    }
                )
            client_metrics_csv = os.path.join(client_dir, f"epoch{epoch}_client{client_idx}_per_attack_metrics.csv")
            pd.DataFrame(rows).to_csv(client_metrics_csv, index=False)
            self.args.logger.info(f"Saved per-attack CSV metrics for client {client_idx} -> {client_metrics_csv}")

            # ===== Non-IID-Dir analytics (mirror PTLAE/SupAE style) =====
            if getattr(self.args, "experiment_type", None) == "non_iid_dir":
                # Collect RE + labels
                try:
                    client_re_list, client_raw_labels = collect_re_and_labels_from_client(client)
                except Exception as e:
                    self.args.logger.warning(f"Failed to collect RE from client {client_idx}: {e}")
                    client_re_list, client_raw_labels = [], []

                # RE distributions
                try:
                    log_re_distributions_from_lists(client_re_list, client_raw_labels, client_dir, epoch, client_id=client_idx)
                except Exception as e:
                    self.args.logger.warning(f"Failed to log RE distributions for client {client_idx}: {e}")

                # Latent norms boxplots
                try:
                    client_z_list, client_z_labels = collect_z_norms_and_labels_from_client(client)
                    log_z_boxplots_from_lists(client_z_list, client_z_labels, client_dir, epoch, client_id=client_idx)
                except Exception as e:
                    self.args.logger.warning(f"Failed to collect/log latent z for client {client_idx}: {e}")

                # Latents + embeddings + histogram(first component)
                try:
                    latents, lat_labels = collect_latents_and_labels_from_client(client, max_samples_per_class=1000)
                    if latents is not None and latents.shape[0] > 0:
                        plot_latent_embedding(latents, lat_labels, client_dir, epoch, client_id=client_idx, method="tsne")
                        try:
                            seen_sets = getattr(self.args, "last_partition_meta", {}) or {}
                            seen_list = seen_sets.get("seen_sets", []) if isinstance(seen_sets, dict) else []
                            seen_for_client = []
                            try:
                                if isinstance(seen_list, list) and client_idx < len(seen_list):
                                    seen_for_client = list(map(int, seen_list[client_idx]))
                            except Exception:
                                seen_for_client = []

                            num_attacks = getattr(self.args, "num_attack_labels", None)
                            if isinstance(num_attacks, int) and num_attacks > 0:
                                attack_labels = list(range(1, num_attacks + 1))
                            else:
                                uniq = sorted(set(int(x) for x in lat_labels if int(x) != 0))
                                attack_labels = uniq

                            out = plot_latent_embedding_non_iid_dir(
                                latents,
                                lat_labels,
                                seen_for_client,
                                attack_label=attack_labels,
                                out_dir=client_dir,
                                epoch=epoch,
                                client_id=client_idx,
                                method="tsne",
                                max_points=2000,
                                random_state=getattr(self.args, "assign_seed", 0),
                            )
                            if out:
                                self.args.logger.info(f"Saved non-iid-dir latent viz for client {client_idx} -> {out}")
                        except Exception:
                            pass

                        try:
                            hist_p = plot_latent_first_component_hist_from_latents(latents, client_dir, epoch, client_id=client_idx)
                            if hist_p:
                                self.args.logger.info(f"Saved client {client_idx} latent-dim0 histogram to {hist_p}")
                            try:
                                first_comp = np.asarray(latents)[:, 0].tolist()
                                global_latent_firstcomp.extend(first_comp)
                            except Exception:
                                pass
                        except Exception:
                            pass
                except Exception as e:
                    self.args.logger.warning(f"Failed to collect/plot latent embedding for client {client_idx}: {e}")

                # Aggregate global/non-iid data
                global_re.extend(client_re_list)
                global_labels.extend(client_raw_labels)
                global_client_idxs.extend([client_idx] * len(client_raw_labels))
                per_client_data[client_idx] = {
                    "re": list(map(float, client_re_list)),
                    "labels": list(map(int, client_raw_labels)),
                }

                # Detailed metrics using predictions
                try:
                    preds, true_labels_raw = collect_predictions_and_labels_from_client(client)

                    pm = getattr(self.args, "last_partition_meta", None) or {}
                    seen_list = pm.get("seen_sets", []) if isinstance(pm, dict) else []
                    seen_for_client = set()
                    if isinstance(seen_list, list) and client_idx < len(seen_list):
                        seen_for_client = set(map(int, seen_list[client_idx]))

                    preds_arr = np.array(preds, dtype=int)
                    labels_arr = np.array(true_labels_raw, dtype=int)

                    mask_normal = labels_arr == 0
                    mask_attack_seen = np.array([lab in seen_for_client and lab != 0 for lab in labels_arr])
                    mask_attack_unseen = np.array([lab not in seen_for_client and lab != 0 for lab in labels_arr])
                    mask_attack_all = labels_arr != 0

                    recall_seen = (preds_arr[mask_attack_seen] == 1).sum() / mask_attack_seen.sum() if mask_attack_seen.sum() > 0 else np.nan
                    recall_unseen = (preds_arr[mask_attack_unseen] == 1).sum() / mask_attack_unseen.sum() if mask_attack_unseen.sum() > 0 else np.nan
                    recall_normal = (preds_arr[mask_normal] == 0).sum() / mask_normal.sum() if mask_normal.sum() > 0 else np.nan

                    predicted_attack = preds_arr == 1
                    if predicted_attack.sum() > 0:
                        correct_attack = predicted_attack & mask_attack_all
                        precision_attack = correct_attack.sum() / predicted_attack.sum()
                    else:
                        precision_attack = np.nan

                    predicted_normal = preds_arr == 0
                    if predicted_normal.sum() > 0:
                        correct_normal = predicted_normal & mask_normal
                        precision_normal = correct_normal.sum() / predicted_normal.sum()
                    else:
                        precision_normal = np.nan

                    detailed_metrics_rows.append(
                        {
                            "Client": f"Client {client_idx}",
                            "Recall_seen": recall_seen * 100 if not np.isnan(recall_seen) else np.nan,
                            "Recall_unseen": recall_unseen * 100 if not np.isnan(recall_unseen) else np.nan,
                            "Recall_normal": recall_normal * 100 if not np.isnan(recall_normal) else np.nan,
                            "Precision_attack": precision_attack * 100 if not np.isnan(precision_attack) else np.nan,
                            "Precision_normal": precision_normal * 100 if not np.isnan(precision_normal) else np.nan,
                        }
                    )
                except Exception as e:
                    self.args.logger.warning(f"Failed detailed metrics for client {client_idx}: {e}")

        # ===== Global non-iid-dir summaries =====
        if getattr(self.args, "experiment_type", None) == "non_iid_dir" and len(global_re) > 0:
            out_dir = os.path.join(
                "logs",
                "re_distributions",
                f"{self.args.model_type}_mc{self.args.num_multi_class_clients}_epoch_{epoch}",
            )

            try:
                log_re_distributions_from_lists(global_re, global_labels, out_dir, epoch, client_id=None)
            except Exception as e:
                self.args.logger.warning(f"Failed to write global RE distributions: {e}")

            try:
                global_z_list = []
                for client in clients:
                    try:
                        z_list, z_labels = collect_z_norms_and_labels_from_client(client)
                        global_z_list.extend(z_list)
                    except Exception:
                        continue
                if len(global_z_list) > 0:
                    log_z_boxplots_from_lists(global_z_list, global_labels, out_dir, epoch, client_id=None)
            except Exception as e:
                self.args.logger.warning(f"Failed to compute/save global latent z distributions: {e}")

            try:
                if len(global_latent_firstcomp) > 0:
                    arr = np.array(global_latent_firstcomp).reshape(-1, 1)
                    global_hist_path = plot_latent_first_component_hist_from_latents(arr, out_dir, epoch, client_id=None)
                    if global_hist_path:
                        self.args.logger.info(f"Saved global latent-dim0 histogram to {global_hist_path}")
            except Exception as e:
                self.args.logger.warning(f"Failed to compute/save global latent-dim0 histogram: {e}")

            try:
                aucs = compute_auc_per_attack_from_flat(global_labels, global_re)
                auc_path = os.path.join(out_dir, f"epoch{epoch}_aucs_per_attack.json")
                aucs_json = {str(int(k)): (None if v is None else float(v)) for k, v in aucs.items()}
                with open(auc_path, "w") as jf:
                    json.dump(aucs_json, jf, indent=2)
                self.args.logger.info(f"Saved per-attack AUCs to {auc_path}")
            except Exception as e:
                self.args.logger.warning(f"Failed to compute/save per-attack AUCs: {e}")

            # Seen/unseen grouping
            try:
                pm = getattr(self.args, "last_partition_meta", None) or {}
                seen_sets = pm.get("seen_sets", []) if isinstance(pm, dict) else []
                attack_label = int(getattr(self.args, "attack_label", 1))
                try:
                    global_threshold = float(np.mean(global_re))
                except Exception:
                    global_threshold = 0.0

                seen_flag = []
                for ci in global_client_idxs:
                    if isinstance(seen_sets, list) and ci < len(seen_sets):
                        try:
                            seen_flag.append(int(attack_label in list(map(int, seen_sets[ci]))))
                        except Exception:
                            seen_flag.append(0)
                    else:
                        seen_flag.append(0)

                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

                def _compute_group_metrics(mask_idxs):
                    if not any(mask_idxs):
                        return {
                            "auc": None,
                            "accuracy": None,
                            "precision": None,
                            "recall": None,
                            "f1": None,
                            "support_attack": 0,
                            "support_benign": 0,
                        }
                    idxs = np.where(np.array(mask_idxs))[0]
                    y = np.array(global_labels)[idxs]
                    scores = np.array(global_re)[idxs]
                    keep_mask = np.isin(y, [0, attack_label])
                    if keep_mask.sum() == 0:
                        return {
                            "auc": None,
                            "accuracy": None,
                            "precision": None,
                            "recall": None,
                            "f1": None,
                            "support_attack": 0,
                            "support_benign": 0,
                        }
                    y2 = y[keep_mask]
                    scores2 = scores[keep_mask]
                    y_bin = (y2 == attack_label).astype(int)
                    try:
                        auc_v = float(roc_auc_score(y_bin, scores2)) if len(np.unique(y_bin)) > 1 else None
                    except Exception:
                        auc_v = None
                    preds = (scores2 > global_threshold).astype(int)
                    try:
                        acc_v = float(accuracy_score(y_bin, preds))
                        prec_v = float(precision_score(y_bin, preds, zero_division=0))
                        rec_v = float(recall_score(y_bin, preds, zero_division=0))
                        f1_v = float(f1_score(y_bin, preds, zero_division=0))
                    except Exception:
                        acc_v = prec_v = rec_v = f1_v = None
                    return {
                        "auc": auc_v,
                        "accuracy": acc_v,
                        "precision": prec_v,
                        "recall": rec_v,
                        "f1": f1_v,
                        "support_attack": int((y2 == attack_label).sum()),
                        "support_benign": int((y2 == 0).sum()),
                    }

                labels_arr = np.array(global_labels)
                seen_flags_arr = np.array(seen_flag)
                mask_attack_samples = np.isin(labels_arr, [0, attack_label])
                seen_group_mask = mask_attack_samples & ((labels_arr == 0) | ((labels_arr == attack_label) & (seen_flags_arr == 1)))
                unseen_group_mask = mask_attack_samples & ((labels_arr == 0) | ((labels_arr == attack_label) & (seen_flags_arr == 0)))

                seen_metrics = _compute_group_metrics(seen_group_mask)
                unseen_metrics = _compute_group_metrics(unseen_group_mask)

                global_seen_payload = {
                    "attack_label": int(attack_label),
                    "global_threshold_used": float(global_threshold),
                    "seen_group": seen_metrics,
                    "unseen_group": unseen_metrics,
                }
                seen_path = os.path.join(out_dir, f"epoch{epoch}_seen_unseen_global_attack{attack_label}.json")
                with open(seen_path, "w") as _jf:
                    json.dump(global_seen_payload, _jf, indent=2)
                self.args.logger.info(f"Saved global seen/unseen metrics to {seen_path}")
            except Exception as e:
                self.args.logger.warning(f"Failed to compute/save non_iid_dir specific metrics: {e}")

            try:
                if detailed_metrics_rows:
                    detailed_df = pd.DataFrame(detailed_metrics_rows)
                    detail_path = os.path.join(out_dir, f"epoch{epoch}_client_seen_unseen_metrics.csv")
                    detailed_df.to_csv(detail_path, index=False)
                    self.args.logger.info(f"Saved per-client seen/unseen metrics to {detail_path}")
            except Exception as e:
                self.args.logger.warning(f"Failed to write per-client seen/unseen metrics: {e}")

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
            rows += "{:<12.1f} {:<20.6f} {:<20.6f} {:<20.6f}\n".format(m, all_avg, benign_avg, poisoned_avg)

        self.args.logger.info(header + rows)

    def log_train_progress(self, epoch, client_idx, client, is_mal):
        self.args.set_train_log_df(
            pd.concat(
                [
                    self.args.get_train_log_df(),
                    pd.DataFrame(
                        [
                            {
                                "epoch": epoch,
                                "client_id": client_idx,
                                "is_mal": is_mal,
                                "train_re": client.recent_re,
                                "train_ce": client.recent_ce,
                                "train_latent_z": client.recent_latent_z,
                                "train_loss": client.recent_train_loss,
                                "val_loss": client.recent_val_loss,
                                "best_val_loss": client.best_loss,
                                "best_epoch": client.best_epoch,
                                "is_training": client.is_training,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        )
