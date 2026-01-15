import os
import torch
import copy
import math
import numpy as np
import pandas as pd
from loguru import logger
from function.utils import (
    average_nn_parameters,
    attention_average_nn_parameters,
    kmeans_cluster_parameters,
    kmeans_cluster_parameters_and_get_min_center,
)
from function.utils.visualize_util import (
    visualize_parameters,
    log_re_distributions_from_lists,
    collect_re_and_labels_from_client,
    collect_z_norms_and_labels_from_client,
    log_z_boxplots_from_lists,
    compute_auc_per_attack_from_flat,
    collect_predictions_and_labels_from_client,
    collect_latents_and_labels_from_client,
    plot_latent_embedding,
    plot_latent_embedding_non_iid_dir,
    plot_latent_first_component_hist_from_latents,
)
import json

class ServerDualLossAE:
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

                # DualLossAE: train trả về (reconstruction_loss, supervised_loss)
                re_loss = client.train(epoch)
                train_loss = re_loss

                list_loss.append(train_loss.item())

                val_loss, threshold_re, threshold_z = client.validate(epoch)
                client.set_recent_metric(re_loss.item(), None, train_loss.item(), val_loss, threshold_re, threshold_z )

                if val_loss < client.best_loss :
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

        # Use a single threshold multiplier consistent with client.test()
        tm = round(float(getattr(self.args, 'threshold_multiplier', 3.0)), 1)
        multipliers = [tm]

        multiplier_auc_all = {m: [] for m in multipliers}
        multiplier_auc_benign = {m: [] for m in multipliers}
        multiplier_auc_poisoned = {m: [] for m in multipliers}
        # accumulate global RE and labels across clients for epoch-level analysis
        global_re = []
        global_re_labels = []
        group_re_by_class = {"normal": [], "seen": [], "unseen": []}
        detailed_metrics_rows = []
        global_latent_firstcomp = []
        global_z_labels = []
        global_client_idxs = []

        for client_idx, client in enumerate(clients):
            self.args.logger.info(
                "Client {} test params: threshold_re (mean={:.6f}, std={:.6f}), best epoch {}".format(
                    client_idx, client.threshold_re[0], client.threshold_re[1], client.best_epoch
                )
            )

            seen_for_client = set()
            if getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
                pm_train = getattr(self.args, 'train_partition_meta', None) or {}
                train_seen_list = pm_train.get('seen_sets', []) if isinstance(pm_train, dict) else []
                if isinstance(train_seen_list, list) and client_idx < len(train_seen_list):
                    seen_for_client = set(map(int, train_seen_list[client_idx]))

            recent_weight_model = client.get_nn_parameters()
            client.update_nn_parameters(client.best_weight_model)

            # Collect per-sample RE and raw labels from client's test set
            try:
                client_re_list, client_raw_labels = collect_re_and_labels_from_client(client)
            except Exception as e:
                self.args.logger.warning(f"Failed to collect RE from client {client_idx}: {e}")
                client_re_list, client_raw_labels = [], []

            # Build out_dir name using model and multi-class setting as requested
            out_dir = os.path.join(
                "logs",
                "re_distributions",
                f"{self.args.model_type}_mc{self.args.num_multi_class_clients}_epoch_{epoch}",
            )
            client_out_dir = os.path.join(out_dir, f"client_{client_idx}")

            # Save per-client RE distributions (CSV + plots)
            try:
                log_re_distributions_from_lists(client_re_list, client_raw_labels, client_out_dir, epoch, client_id=client_idx)
            except Exception as e:
                self.args.logger.warning(f"Failed to log RE distributions for client {client_idx}: {e}")

            # Also collect & log latent (z) boxplots (L2 norms) per-client and per-attack
            # try:
                # client_z_list, client_z_labels = collect_z_norms_and_labels_from_client(client)
                # log_z_boxplots_from_lists(client_z_list, client_z_labels, client_out_dir, epoch, client_id=client_idx)
            # except Exception as e:
            #     self.args.logger.warning(f"Failed to collect/log latent z for client {client_idx}: {e}")

            # accumulate for global aggregation
            global_re.extend(client_re_list)
            global_re_labels.extend(client_raw_labels)

            if getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
                for r, lbl in zip(client_re_list, client_raw_labels):
                    lbl_int = int(lbl)
                    if lbl_int == 0:
                        group_re_by_class["normal"].append(r)
                    elif lbl_int in seen_for_client:
                        group_re_by_class["seen"].append(r)
                    else:
                        group_re_by_class["unseen"].append(r)

            acc_list, precision_list, recall_list, f1_list, auc_list = client.test()

            client.update_nn_parameters(recent_weight_model)

            for m, acc, precision, recall, f1, auc in zip(multipliers, acc_list, precision_list, recall_list, f1_list, auc_list):

                if m == 0.0:
                    self.args.logger.info(
                        f"[Client {client_idx}] Multiplier {m:.1f}: ACC={acc:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
                    )

                # Append to test log DataFrame safely, avoiding concat with None
                existing_df = self.args.get_test_log_df()
                new_row = pd.DataFrame([
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
                ])
                if existing_df is None or (isinstance(existing_df, pd.DataFrame) and existing_df.empty):
                    self.args.set_test_log_df(new_row)
                else:
                    self.args.set_test_log_df(pd.concat([existing_df, new_row], ignore_index=True))

                multiplier_auc_all[m].append(auc)
                if client_idx in poisoned_workers:
                    multiplier_auc_poisoned[m].append(auc)
                else:
                    multiplier_auc_benign[m].append(auc)

            # Detailed metrics for non_iid_dir using client predictions
            if getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
                try:
                    preds, true_labels_raw = collect_predictions_and_labels_from_client(client)

                    pm = getattr(self.args, 'last_partition_meta', None) or {}
                    seen_list = pm.get('seen_sets', []) if isinstance(pm, dict) else []
                    seen_for_client_metrics = set(seen_for_client)
                    if not seen_for_client_metrics and isinstance(seen_list, list) and client_idx < len(seen_list):
                        seen_for_client_metrics = set(map(int, seen_list[client_idx]))

                    preds_arr = np.array(preds, dtype=int)
                    labels_arr = np.array(true_labels_raw, dtype=int)

                    mask_normal = labels_arr == 0
                    mask_attack_seen = np.array([lab in seen_for_client_metrics and lab != 0 for lab in labels_arr])
                    mask_attack_unseen = np.array([lab not in seen_for_client_metrics and lab != 0 for lab in labels_arr])
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

            # Per-client latent embedding (t-SNE) and non-iid-dir specialized embedding
            latents, lat_labels = collect_latents_and_labels_from_client(client, max_samples_per_class=1000)
            if latents is not None and latents.shape[0] > 0:
                plot_latent_embedding(latents, lat_labels, client_out_dir, epoch, client_id=client_idx, method='tsne')
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
                        out_dir=client_out_dir,
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
                # Grab latent encoding directly to avoid unpacking a scalar loss
                try:
                    with torch.no_grad():
                        latent, _ = client.net(input)
                        if latent.numel() > 0:
                            z_val = latent.view(-1)[0]
                            global_latent_firstcomp.append(float(z_val.item()))
                            global_client_idxs.append(client_idx)
                except Exception as e:
                    self.args.logger.warning(f"Failed to record latent for client {client_idx}: {e}")

        out_dir = os.path.join(
            "logs",
            "re_distributions",
            f"{self.args.model_type}_mc{self.args.num_multi_class_clients}_epoch_{epoch}",
        )

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

            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, f"epoch{epoch}_detailed_metrics.csv")
            df_det.to_csv(csv_path, index=False, encoding="utf-8-sig")
            self.args.logger.info(f"Saved detailed seen/unseen metrics to {csv_path}")

        # Global histogram latent-dim0 (z) for non_iid_dir
        if getattr(self.args, 'experiment_type', None) == 'non_iid_dir' and len(global_latent_firstcomp) > 0:
            os.makedirs(out_dir, exist_ok=True)
            arr = np.array(global_latent_firstcomp).reshape(-1, 1)
            global_hist_path = plot_latent_first_component_hist_from_latents(arr, out_dir, epoch, client_id=None)
            if global_hist_path:
                self.args.logger.info(f"Saved global latent-dim0 histogram to {global_hist_path}")

        # After per-client collection, produce global RE plots and per-attack AUCs
        if 'global_re' in locals() and len(global_re) > 0:
            try:
                summary, grouped = log_re_distributions_from_lists(global_re, global_re_labels, out_dir, epoch, client_id=None)
            except Exception as e:
                self.args.logger.warning(f"Failed to write global RE distributions: {e}")

            # global latent z norms + boxplots
            try:
                global_z_list = []
                # collect latent z norms across clients
                for client in clients:
                    try:
                        z_list, z_labels = collect_z_norms_and_labels_from_client(client)
                        global_z_list.extend(z_list)
                        global_z_labels.extend(z_labels)
                    except Exception:
                        # skip clients that fail
                        continue
                # if len(global_z_list) > 0:
                #     log_z_boxplots_from_lists(global_z_list, global_z_labels, out_dir, epoch, client_id=None)
            except Exception as e:
                self.args.logger.warning(f"Failed to compute/save global latent z distributions: {e}")

            try:
                aucs = compute_auc_per_attack_from_flat(global_re_labels, global_re)
                auc_path = os.path.join(out_dir, f"epoch{epoch}_aucs_per_attack.json")
                with open(auc_path, "w") as jf:
                    json.dump(aucs, jf, indent=2)
                self.args.logger.info(f"Saved per-attack AUCs to {auc_path}")
            except Exception as e:
                self.args.logger.warning(f"Failed to compute/save per-attack AUCs: {e}")

        if getattr(self.args, 'experiment_type', None) == 'non_iid_dir':
            rows = []
            for group_name, values in group_re_by_class.items():
                arr = np.array(values, dtype=float)
                rows.append({
                    "epoch": epoch,
                    "group": group_name,
                    "count": int(arr.size),
                    "mean_mse": float(np.mean(arr)) if arr.size > 0 else np.nan,
                    "std_mse": float(np.std(arr)) if arr.size > 0 else np.nan,
                })

            mse_csv = os.path.join(out_dir, "mse_seen_unseen.csv")
            header = not os.path.exists(mse_csv)
            pd.DataFrame(rows).to_csv(mse_csv, mode="a", header=header, index=False)
            self.args.logger.info(f"Saved MSE summary (normal/seen/unseen) to {mse_csv}")

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
