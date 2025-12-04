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
from function.utils.visualize_util import visualize_parameters

class ServerSAE:
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

                # SAE: train trả về (re_loss, latent_z_loss)
                re_loss, latent_z_loss = client.train(epoch)
                train_loss = re_loss + self.args.coef_shrink_ae * latent_z_loss

                list_loss.append(train_loss.item())

                val_loss, threshold_re, threshold_z = client.validate(epoch)
                client.set_recent_metric(re_loss.item(), latent_z_loss.item(), train_loss.item(), val_loss, threshold_re, threshold_z)

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

                # 💾 Lưu mô hình mỗi 10 epoch
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

        multipliers = np.arange(0.0, 5.1, 0.2)

        multiplier_auc_all = {m: [] for m in multipliers}
        multiplier_auc_benign = {m: [] for m in multipliers}
        multiplier_auc_poisoned = {m: [] for m in multipliers}

        for client_idx, client in enumerate(clients):
            self.args.logger.info(
                "Client {} test params:  threshold_z (mean={:.6f}, std={:.6f}), best epoch {}".format(
                    client_idx, client.threshold_z[0], client.threshold_z[1], client.best_epoch
                )
            )

            recent_weight_model = client.get_nn_parameters()
            client.update_nn_parameters(client.best_weight_model)

            acc_list, precision_list, recall_list, f1_list, auc_list = client.test()

            client.update_nn_parameters(recent_weight_model)

            for i, m in enumerate(multipliers):
                acc, precision, recall, f1, auc = acc_list[i], precision_list[i], recall_list[i], f1_list[i], auc_list[i]

                if m == 0.0:
                    self.args.logger.info(
                        f"[Client {client_idx}] Multiplier {m:.1f}: ACC={acc:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
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
