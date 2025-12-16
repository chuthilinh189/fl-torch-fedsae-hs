import os
import torch
import copy
import math
import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score, recall_score
from loguru import logger
from function.utils import (
    average_nn_parameters,
    attention_average_nn_parameters,
    kmeans_cluster_parameters,
    kmeans_cluster_parameters_and_get_min_center,
)
from function.utils.visualize_util import visualize_parameters
from function.utils.visualize_util import (
    plot_latent_first_component_hist_from_latents,
    collect_latents_and_labels_from_client,
    collect_predictions_and_labels_from_client,
    plot_latent_embedding,
    plot_latent_embedding_non_iid_dir,
)

class ServerSupAE:
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

                # SupAE: train trả về supervised loss
                supervised_loss = client.train(epoch)
                list_loss.append(supervised_loss.item())

                val_loss, threshold_re, threshold_z = client.validate(epoch)
                client.set_recent_metric(None, None, supervised_loss.item(), val_loss, threshold_re, threshold_z)

                if val_loss < client.best_loss:
                    best_weight_model = copy.deepcopy(client.get_nn_parameters())
                    client.set_best_ckpt(val_loss, epoch, threshold_re, threshold_z, best_weight_model)
                    self.args.logger.info(
                        "Client {} gets new best val_loss at epoch #{}: {:.6f}",
                        str(client_idx), str(client.best_epoch), client.best_loss
                    )
                elif client.best_epoch + self.args.es_offset <= epoch:
                    client.set_training_status(False)
                    self.args.logger.info(
                        "Client {} early stopped at epoch #{}", str(client_idx), str(epoch)
                    )

                # 💾 Lưu model mỗi 10 epoch
                if client.is_training and epoch % 10 == 0:
                    save_dir = f"saved_models/{self.args.model_type}/{self.args.num_multi_class_clients}/{self.args.aggregation_type}/{self.args.dataset}/"
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(save_dir, f"epoch_{epoch}_client_{client_idx}.pt")
                    torch.save(client.net.state_dict(), model_path)
                    self.args.logger.info(
                        f"Saved model for client {client_idx} at epoch {epoch} -> {model_path}"
                    )

                self.log_train_progress(epoch, client_idx, client, client_idx in poisoned_workers)

        self.args.logger.info(
            "{} clients still training at epoch #{}", str(len(list_client_training)), str(epoch)
        )

        if len(list_client_training) > 0:
            parameters = [
                clients[client_idx].get_nn_parameters()
                for client_idx in list_client_training
            ]

            if self.args.aggregation_type == "cluster" and len(list_client_training) > 1:
                client_weights = parameters
                cluster_result = kmeans_cluster_parameters(client_weights, n_clusters=2)
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

        # Train-time latent[0] histograms every N epochs for non_iid_dir, using TEST latents (align with ServerPTL)
        if (
            getattr(self.args, 'experiment_type', None) == 'non_iid_dir'
            and (epoch < 10 or (epoch % 10 == 0 and epoch < 100) or epoch % 100 == 0)
        ):
            out_dir = os.path.join(
                "logs",
                "re_distributions",
                f"{self.args.model_type}_mc{self.args.num_multi_class_clients}",
                "train_hist",
            )
            os.makedirs(out_dir, exist_ok=True)

            global_latent_firstcomp_train = []
            for client_idx, client in enumerate(clients):
                # Collect TEST latent z per sample
                z_list = []
                for input, _ in client.test_data_loader:
                    input = input.to(client.device)
                    dummy_label = torch.zeros(input.size(0), dtype=torch.long, device=input.device)
                    _, z = client.calculate_loss(input, dummy_label)
                    z_list.append(float(z.item()))

                if len(z_list) > 0:
                    client_out_dir = os.path.join(out_dir, f"client_{client_idx}")
                    os.makedirs(client_out_dir, exist_ok=True)
                    arr = np.array(z_list, dtype=float).reshape(-1, 1)
                    hist_p = plot_latent_first_component_hist_from_latents(arr, client_out_dir, epoch, client_id=client_idx)
                    if hist_p:
                        self.args.logger.info(f"[Train] Saved client {client_idx} latent-dim0 histogram to {hist_p}")
                    global_latent_firstcomp_train.extend(z_list)

            # Global train histogram
            if len(global_latent_firstcomp_train) > 0:
                arr = np.array(global_latent_firstcomp_train, dtype=float).reshape(-1, 1)
                global_hist_path = plot_latent_first_component_hist_from_latents(arr, out_dir, epoch, client_id=None)
                if global_hist_path:
                    self.args.logger.info(f"[Train] Saved global latent-dim0 histogram to {global_hist_path}")

        return len(list_client_training) == 0

    def aggregate_parameters(self, parameters, list_loss):
        if self.args.aggregation_type == "average":
            return average_nn_parameters(parameters)

        elif self.args.aggregation_type == "attention":
            np_list_loss = np.asarray(list_loss, dtype=np.float32)
            loss_sum = np.sum(np_list_loss)
            list_weight_loss = np.log(loss_sum / np_list_loss)
            weight_loss_sum = np.sum(list_weight_loss)
            list_aggregation_coef = list_weight_loss / weight_loss_sum

            if len(parameters) == 1:
                list_aggregation_coef = [1.0]

            return attention_average_nn_parameters(parameters, list_aggregation_coef)

        elif self.args.aggregation_type == "split":
            k = 0.2
            num_keep = math.ceil(len(list_loss) * (1 - k))
            sorted_losses_params = sorted(zip(list_loss, parameters), key=lambda x: x[0])
            selected_params = [p for l, p in sorted_losses_params[:num_keep]]
            return average_nn_parameters(selected_params)

        elif self.args.aggregation_type == "kmean":
            clustered_params = kmeans_cluster_parameters_and_get_min_center(parameters, list_loss, 2)
            return average_nn_parameters(clustered_params)

        else:
            raise ValueError(f"Unsupported aggregation type: {self.args.aggregation_type}")

    def test_on_clients(self, epoch, clients, poisoned_workers):
        self.args.logger.info("Testing {} model at epoch #{}", self.args.model_type, str(epoch))

        multipliers = np.arange(0.0, 0.1, 0.2)

        multiplier_auc_all = {m: [] for m in multipliers}
        multiplier_auc_benign = {m: [] for m in multipliers}
        multiplier_auc_poisoned = {m: [] for m in multipliers}

        # accumulate global labels and z-first-component across clients for epoch-level analysis
        global_labels = []
        global_latent_firstcomp = []
        global_client_idxs = []
        detailed_metrics_rows = []

        for client_idx, client in enumerate(clients):
            self.args.logger.info(
            "Client {} test params:  threshold_z (mean={:.6f}, std={:.6f}), best epoch {}".format(
                client_idx, client.threshold_z[0], client.threshold_z[1], client.best_epoch
            )
        )


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
                                    "threshold_multiplier": 0,
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

            multiplier_auc_all[0].append(auc)
            if client_idx in poisoned_workers:
                multiplier_auc_poisoned[0].append(auc)
            else:
                multiplier_auc_benign[0].append(auc)

            # Detailed metrics for non_iid_dir using client predictions
            if getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
                try:
                    preds, true_labels_raw = collect_predictions_and_labels_from_client(client)

                    pm = getattr(self.args, 'last_partition_meta', None) or {}
                    seen_list = pm.get('seen_sets', []) if isinstance(pm, dict) else []
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

                    detailed_metrics_rows.append({
                        "Client": f"Client {client_idx}",
                        "Recall_seen": recall_seen * 100 if not np.isnan(recall_seen) else np.nan,
                        "Recall_unseen": recall_unseen * 100 if not np.isnan(recall_unseen) else np.nan,
                        "Recall_normal": recall_normal * 100 if not np.isnan(recall_normal) else np.nan,
                        "Precision_attack": precision_attack * 100 if not np.isnan(precision_attack) else np.nan,
                        "Precision_normal": precision_normal * 100 if not np.isnan(precision_normal) else np.nan,
                    })
                except Exception as e:
                    self.args.logger.warning(f"Failed detailed metrics for client {client_idx}: {e}")

            # Per-attack metrics CSV (benign vs attack k) using client's threshold_z mean
            out_dir = os.path.join(
                "logs",
                "re_distributions",
                f"{self.args.model_type}_mc{self.args.num_multi_class_clients}_epoch_{epoch}",
            )
            client_dir = os.path.join(out_dir, f"client_{client_idx}")
            os.makedirs(client_dir, exist_ok=True)
            th_z_mean = float(client.threshold_z[0]) if isinstance(client.threshold_z, (list, tuple)) else float(client.threshold_z)
            classif = client.test_by_attack_type_full(None, th_z_mean, verbose=False)
            # collect per-client z scores and labels to compute per-attack AUCs (using z as score)
            z_scores = []
            lbls = []
            for input, label in client.test_data_loader:
                input, label = input.to(client.device), label.to(client.device)
                dummy_label = torch.zeros(input.size(0), dtype=torch.long, device=input.device)
                _, z = client.calculate_loss(input, dummy_label)
                z_scores.append(float(z.item()))
                lbls.append(int(label.item()))
            # compute per-attack AUCs: benign (0) vs attack k, using z_scores
            per_attack_auc = {}
            labels_arr = np.array(lbls, dtype=int)
            scores_arr = np.array(z_scores, dtype=float)
            for atk_type in sorted(set(labels_arr.tolist())):
                if atk_type == 0:
                    continue
                mask = np.isin(labels_arr, [0, atk_type])
                if mask.sum() == 0:
                    per_attack_auc[atk_type] = None
                    continue
                y_bin = (labels_arr[mask] == atk_type).astype(int)
                s = scores_arr[mask]
                per_attack_auc[atk_type] = float(roc_auc_score(y_bin, s)) if len(np.unique(y_bin)) > 1 else None
            rows = []
            for atk_id, metrics in classif.items():
                rows.append({
                    'epoch': int(epoch),
                    'client_id': int(client_idx),
                    'attack_type': int(atk_id),
                    'auc': float(per_attack_auc.get(int(atk_id), 0.0) or 0.0),
                    'accuracy': float(metrics.get('acc', 0.0)),
                    'precision': float(metrics.get('precision', 0.0)),
                    'recall': float(metrics.get('recall', 0.0)),
                    'f1': float(metrics.get('f1-score', 0.0)),
                    'support': int(metrics.get('support', 0)),
                })
            client_metrics_csv = os.path.join(client_dir, f"epoch{epoch}_client{client_idx}_per_attack_metrics.csv")
            pd.DataFrame(rows).to_csv(client_metrics_csv, index=False)
            self.args.logger.info(f"Saved per-attack CSV metrics for client {client_idx} -> {client_metrics_csv}")

            # Per-client latent embedding (t-SNE) and non-iid-dir specialized embedding
            latents, lat_labels = collect_latents_and_labels_from_client(client, max_samples_per_class=1000)
            if latents is not None and latents.shape[0] > 0:
                # Standard t-SNE embedding
                plot_latent_embedding(latents, lat_labels, client_dir, epoch, client_id=client_idx, method='tsne')
                # For non-iid-dir experiments, also produce the specialized 3-color embedding
                if getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
                    pm_train = getattr(self.args, 'train_partition_meta', None) or {}
                    seen_list = pm_train.get('seen_sets', []) if isinstance(pm_train, dict) else []
                    seen_for_client = []
                    if isinstance(seen_list, list) and client_idx < len(seen_list):
                        seen_for_client = list(map(int, seen_list[client_idx]))
                    num_attacks = getattr(self.args, 'num_attack_labels', None)
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
                        method='tsne',
                        max_points=1000,
                        random_state=getattr(self.args, 'assign_seed', 0),
                    )
                    if out:
                        self.args.logger.info(f"Saved non-iid-dir latent viz for client {client_idx} -> {out}")

            # accumulate global labels and latent first component (z)
            for input, label in client.test_data_loader:
                input, label = input.to(client.device), label.to(client.device)
                dummy_label = torch.zeros(input.size(0), dtype=torch.long, device=input.device)
                _, z = client.calculate_loss(input, dummy_label)
                global_latent_firstcomp.append(float(z.item()))
                global_labels.append(int(label.item()))
                global_client_idxs.append(client_idx)

        # After per-client collection: for non_iid_dir, global histogram and per-attack recall
        out_dir = os.path.join(
            "logs",
            "re_distributions",
            f"{self.args.model_type}_mc{self.args.num_multi_class_clients}_epoch_{epoch}",
        )
        os.makedirs(out_dir, exist_ok=True)

        # Export detailed metrics CSV for non_iid_dir
        if detailed_metrics_rows and getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
            df_det = pd.DataFrame(detailed_metrics_rows)
            global_recall_seen = np.nanmean(df_det["Recall_seen"].values)
            global_recall_unseen = np.nanmean(df_det["Recall_unseen"].values)
            global_recall_normal = np.nanmean(df_det["Recall_normal"].values)
            global_precision_attack = np.nanmean(df_det["Precision_attack"].values)
            global_precision_normal = np.nanmean(df_det["Precision_normal"].values)

            global_row = {
                "Client": "All",
                "Recall_seen": global_recall_seen,
                "Recall_unseen": global_recall_unseen,
                "Recall_normal": global_recall_normal,
                "Precision_attack": global_precision_attack,
                "Precision_normal": global_precision_normal,
            }
            df_det = pd.concat([df_det, pd.DataFrame([global_row])], ignore_index=True)

            csv_path = os.path.join(out_dir, f"epoch{epoch}_detailed_metrics.csv")
            df_det.to_csv(csv_path, index=False, encoding="utf-8-sig")
            self.args.logger.info(f"Saved detailed seen/unseen metrics to {csv_path}")

        # Global histogram latent-dim0 (z) for non_iid_dir
        if getattr(self.args, 'experiment_type', None) == 'non_iid_dir' and len(global_latent_firstcomp) > 0:
            arr = np.array(global_latent_firstcomp).reshape(-1, 1)
            global_hist_path = plot_latent_first_component_hist_from_latents(arr, out_dir, epoch, client_id=None)
            if global_hist_path:
                self.args.logger.info(f"Saved global latent-dim0 histogram to {global_hist_path}")

        # Global recall per attack (benign vs k) using z-threshold=0.5
        if len(global_labels) > 0:
            labels_arr = np.array(global_labels, dtype=int)
            recall_dict = {}
            for atk_type in sorted(set(labels_arr.tolist())):
                if atk_type == 0:
                    continue
                y_true_bin_all = []
                y_pred_bin_all = []
                for cidx, client in enumerate(clients):
                    for input, label in client.test_data_loader:
                        input, label = input.to(client.device), label.to(client.device)
                        dummy_label = torch.zeros(input.size(0), dtype=torch.long, device=input.device)
                        _, z = client.calculate_loss(input, dummy_label)
                        lab = int(label.item())
                        if lab in (0, atk_type):
                            y_true_bin_all.append(1 if lab == atk_type else 0)
                            y_pred_bin_all.append(1 - int(float(z.item()) <= 0.5))
                recall_dict[str(int(atk_type))] = (float(recall_score(y_true_bin_all, y_pred_bin_all, zero_division=0)) if len(y_true_bin_all) > 0 else None)
            recall_path = os.path.join(out_dir, f"epoch{epoch}_recall_per_attack.json")
            with open(recall_path, 'w') as jf:
                json.dump(recall_dict, jf, indent=2)
            self.args.logger.info(f"Saved global per-attack recall to {recall_path}")

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
                                "train_latent_z": client.recent_latent_z,
                                "train_loss": client.recent_train_loss,
                                "val_loss": client.recent_val_loss,
                                "threshold_re": client.recent_threshold_re,
                                "threshold_z": client.recent_threshold_z,
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
