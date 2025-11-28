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

    def train_on_clients(self, epoch, clients, poisoned_workers, pdf_writer):
        self.args.logger.info("Training {} model epoch #{}", self.args.model_type, str(epoch))

        list_loss = []
        list_client_training = []
        proto_stats_list = []

        for client_idx, client in enumerate(clients):
            if client.is_training:
                list_client_training.append(client_idx)

                # client.train returns (avg_loss, local_proto_stats)
                result = client.train(epoch, pdf_writer)
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

        for client_idx, client in enumerate(clients):
            self.args.logger.info(
                "Client {} test params: threshold_re (mean={:.6f}, std={:.6f}), best epoch {}".format(
                    client_idx, client.threshold_re[0], client.threshold_re[1], client.best_epoch
                )
            )

            recent_weight_model = client.get_nn_parameters()
            client.update_nn_parameters(client.best_weight_model)

            # Collect per-sample RE and raw labels from client's test set
            try:
                client_re_list, client_raw_labels = collect_re_and_labels_from_client(client)
            except Exception as e:
                self.args.logger.warning(f"Failed to collect RE from client {client_idx}: {e}")
                client_re_list, client_raw_labels = [], []

            # Build out_dir name using model and multi-class setting
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

            # Also collect & log latent (z) boxplots (L2 norms) per-client
            try:
                client_z_list, client_z_labels = collect_z_norms_and_labels_from_client(client)
                log_z_boxplots_from_lists(client_z_list, client_z_labels, client_out_dir, epoch, client_id=client_idx)
            except Exception as e:
                self.args.logger.warning(f"Failed to collect/log latent z for client {client_idx}: {e}")

            # Collect full latent vectors and produce t-SNE/UMAP plots with prototype overlay
            try:
                latents, lat_labels = collect_latents_and_labels_from_client(client, max_samples_per_class=1000)
                if latents is not None and latents.shape[0] > 0:
                    try:
                        plot_latent_embedding(latents, lat_labels, client_out_dir, epoch, client_id=client_idx, proto_z0=self.proto_z0, proto_z1=self.proto_z1, method='tsne')
                    except Exception:
                        pass
                    try:
                        # attempt UMAP if available; function will fall back if not
                        plot_latent_embedding(latents, lat_labels, client_out_dir, epoch, client_id=client_idx, proto_z0=self.proto_z0, proto_z1=self.proto_z1, method='umap')
                    except Exception:
                        pass
            except Exception as e:
                self.args.logger.warning(f"Failed to collect/plot latent embedding for client {client_idx}: {e}")

            # accumulate for global aggregation
            global_re.extend(client_re_list)
            global_labels.extend(client_raw_labels)

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
            try:
                summary, grouped = log_re_distributions_from_lists(global_re, global_labels, out_dir, epoch, client_id=None)
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
                    except Exception:
                        continue
                if len(global_z_list) > 0:
                    log_z_boxplots_from_lists(global_z_list, global_labels, out_dir, epoch, client_id=None)
            except Exception as e:
                self.args.logger.warning(f"Failed to compute/save global latent z distributions: {e}")

            try:
                aucs = compute_auc_per_attack_from_flat(global_labels, global_re)
                auc_path = os.path.join(out_dir, f"epoch{epoch}_aucs_per_attack.json")
                with open(auc_path, "w") as jf:
                    json.dump(aucs, jf, indent=2)
                self.args.logger.info(f"Saved per-attack AUCs to {auc_path}")
            except Exception as e:
                self.args.logger.warning(f"Failed to compute/save per-attack AUCs: {e}")

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
            rp0 = getattr(client, 'recent_proto_z0', None)
            rp1 = getattr(client, 'recent_proto_z1', None)
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
