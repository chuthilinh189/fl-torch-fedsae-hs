import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

try:  # optional UMAP support
    import umap

    _HAS_UMAP = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_UMAP = False

from function.utils import (
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


class ServerFedGH:
    def __init__(self, args):
        self.args = args
        self.server_device = torch.device("cuda:0") if torch.cuda.is_available() and self.args.cuda else torch.device("cpu")
        self.global_predictor = None

    def get_predictor(self, clients):
        if self.global_predictor is None:
            self.global_predictor = copy.deepcopy(clients[0].net.predictor).to(self.server_device)
        return self.global_predictor

    def aggregate_prototypes(self, proto_list):
        agg = {}
        for proto in proto_list:
            for lbl, (vec_sum, cnt) in proto.items():
                if lbl not in agg:
                    agg[lbl] = [torch.tensor(vec_sum, device=self.server_device), cnt]
                else:
                    agg[lbl][0] += torch.tensor(vec_sum, device=self.server_device)
                    agg[lbl][1] += cnt
        dataset = []
        for lbl, (vec_sum, cnt) in agg.items():
            mean_vec = vec_sum / max(cnt, 1)
            dataset.append((mean_vec, torch.tensor(lbl, dtype=torch.long, device=self.server_device)))
        return dataset

    def train_predictor_from_prototypes(self, clients, proto_dataset):
        predictor = self.get_predictor(clients)
        if not proto_dataset:
            return
        predictor.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(predictor.parameters(), lr=self.args.learning_rate)
        for _ in range(1):
            for feat, lbl in proto_dataset:
                optimizer.zero_grad()
                logits = predictor(feat.unsqueeze(0))
                loss = criterion(logits, lbl.unsqueeze(0))
                loss.backward()
                optimizer.step()

    def broadcast_predictor(self, clients):
        predictor_state = copy.deepcopy(self.global_predictor.state_dict())
        for client in clients:
            client.set_predictor_parameters(predictor_state)


    def _plot_latent_with_prototypes(
        self,
        latents,
        labels,
        proto_stats,
        out_dir,
        epoch,
        client_id=None,
        method="tsne",
        max_points=1000,
        random_state=42,
    ):
        """Plot t-SNE/UMAP with prototype overlay by embedding latents + prototype means together."""
        os.makedirs(out_dir, exist_ok=True)
        if latents is None or latents.shape[0] == 0:
            return None

        labels_arr = np.array(labels)

        # sample latents for plotting
        N = latents.shape[0]
        if N > max_points:
            kept = []
            unique_lbls = np.unique(labels_arr)
            k_per = max_points // max(1, len(unique_lbls))
            for lbl in unique_lbls:
                idxs = np.where(labels_arr == lbl)[0]
                if len(idxs) > k_per:
                    chosen = np.random.choice(idxs, size=k_per, replace=False).tolist()
                else:
                    chosen = idxs.tolist()
                kept.extend(chosen)
            kept = sorted(kept)
            latents_plot = latents[kept]
            labels_plot = labels_arr[kept]
        else:
            latents_plot = latents
            labels_plot = labels_arr

        # compute prototype means
        proto_vecs = []
        proto_labels = []
        if proto_stats:
            for lbl, (vec_sum, cnt) in proto_stats.items():
                if cnt <= 0:
                    continue
                try:
                    vec = np.array(vec_sum) / float(cnt)
                    vec = np.reshape(vec, (-1,))
                    proto_vecs.append(vec)
                    proto_labels.append(int(lbl))
                except Exception:
                    continue

        combined = latents_plot
        combined_labels = labels_plot
        is_proto_mask = None
        if proto_vecs:
            proto_arr = np.stack(proto_vecs, axis=0)
            combined = np.concatenate([latents_plot, proto_arr], axis=0)
            combined_labels = np.concatenate([labels_plot, np.array(proto_labels)])
            is_proto_mask = np.zeros(combined.shape[0], dtype=bool)
            is_proto_mask[-len(proto_vecs):] = True

        scaler = StandardScaler()
        combined_std = scaler.fit_transform(combined)

        # Guard against degenerate embeddings (all points identical or zero variance) which can
        # trigger divide-by-zero warnings in PCA/TSNE.
        if combined_std.shape[0] < 2 or np.allclose(combined_std, combined_std[0]) or np.allclose(
            np.std(combined_std, axis=0), 0
        ):
            emb_points = np.zeros((combined_std.shape[0], 2), dtype=np.float32)
        else:
            if method == "umap" and _HAS_UMAP:
                reducer = umap.UMAP(n_components=2, random_state=random_state)
            else:
                reducer = TSNE(n_components=2, init="pca", random_state=random_state)

            try:
                emb_points = reducer.fit_transform(combined_std)
                if np.allclose(np.std(emb_points, axis=0), 0):
                    emb_points = np.zeros_like(emb_points)
            except Exception:
                emb_points = np.zeros((combined_std.shape[0], 2), dtype=np.float32)

        fname_base = f"epoch{epoch}"
        if client_id is not None:
            fname_base += f"_client{client_id}"

        import matplotlib.pyplot as plt  # local import to avoid global dependency if headless

        plt.figure(figsize=(7, 6))
        unique_lbls = sorted(np.unique(labels_plot))
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_lbls))))
        for i, lbl in enumerate(unique_lbls):
            mask = combined_labels == lbl
            if is_proto_mask is not None:
                mask = mask & (~is_proto_mask)
            plt.scatter(emb_points[mask, 0], emb_points[mask, 1], s=8, c=[colors[i]], label=str(lbl), alpha=0.7)

        if is_proto_mask is not None:
            plt.scatter(
                emb_points[is_proto_mask, 0],
                emb_points[is_proto_mask, 1],
                marker="*",
                s=120,
                c="k",
                edgecolors="white",
                linewidths=0.8,
                label="prototypes",
            )

        plt.legend(markerscale=2)
        plt.title(
            f"Latent embedding ({method}) with prototypes - epoch {epoch}"
            + (f" client {client_id}" if client_id is not None else "")
        )
        plt.xlabel("dim1")
        plt.ylabel("dim2")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{fname_base}_{method}_proto.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        return out_path

    def train_on_clients(self, epoch, clients, poisoned_workers):
        self.args.logger.info("Training {} model epoch #{}", self.args.model_type, str(epoch))

        proto_list = []
        list_loss = []
        list_client_training = []

        for client_idx, client in enumerate(clients):
            if not client.is_training:
                continue

            list_client_training.append(client_idx)
            total_loss = client.train(epoch)
            list_loss.append(float(total_loss.detach().cpu().item()))

            val_loss = client.validate(epoch)
            client.set_recent_metric(client.recent_re, client.recent_ce, client.recent_train_loss, val_loss)

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

            proto_stats = client.collect_prototypes()
            proto_list.append(proto_stats)

        self.args.logger.info("{} clients still training at epoch #{}", str(len(list_client_training)), str(epoch))

        if len(list_client_training) == 0:
            return True

        avg_loss = sum(list_loss) / len(list_loss) if list_loss else 0.0
        self.args.logger.info(f"Avg client loss: {avg_loss:.4f}")

        proto_dataset = self.aggregate_prototypes(proto_list)
        self.train_predictor_from_prototypes(clients, proto_dataset)
        if self.global_predictor is not None:
            self.broadcast_predictor(clients)

        return False

    def test_on_clients(self, epoch, clients, poisoned_workers):
        self.args.logger.info("Testing {} model at epoch #{}", self.args.model_type, str(epoch))

        acc_list, precision_list, recall_list, f1_list, roc_list = [], [], [], [], []
        acc_class_list = []
        best_f1, best_client_idx = -1, -1

        # non-iid-dir support
        global_re, global_labels, global_client_idxs = [], [], []
        global_latent_firstcomp = []
        detailed_metrics_rows = []
        per_client_data = {}

        for idx, client in enumerate(clients):
            self.args.logger.info("Client {} test params: best epoch {}", idx, client.best_epoch)

            if client.best_epoch == -1:
                client.set_best_ckpt(client.best_loss, 0, client.best_weight_model)

            recent_weight_model = client.get_nn_parameters()
            client.update_nn_parameters(client.best_weight_model)

            # === non-iid-dir analytics & visualizations ===
            out_dir = os.path.join(
                "logs",
                "re_distributions",
                f"{self.args.model_type}_mc{self.args.num_multi_class_clients}_epoch_{epoch}",
            )
            client_dir = os.path.join(out_dir, f"client_{idx}")
            os.makedirs(client_dir, exist_ok=True)

            # RE distributions and z boxplots
            try:
                client_re_list, client_raw_labels = collect_re_and_labels_from_client(client)
            except Exception as e:
                self.args.logger.warning(f"Failed to collect RE from client {idx}: {e}")
                client_re_list, client_raw_labels = [], []

            try:
                log_re_distributions_from_lists(client_re_list, client_raw_labels, client_dir, epoch, client_id=idx)
            except Exception as e:
                self.args.logger.warning(f"Failed to log RE distributions for client {idx}: {e}")

            try:
                client_z_list, client_z_labels = collect_z_norms_and_labels_from_client(client)
                log_z_boxplots_from_lists(client_z_list, client_z_labels, client_dir, epoch, client_id=idx)
            except Exception as e:
                self.args.logger.warning(f"Failed to collect/log latent z for client {idx}: {e}")

            # Latent embeddings with prototype overlay
            try:
                latents, lat_labels = collect_latents_and_labels_from_client(client, max_samples_per_class=1000)
                if latents is not None and latents.shape[0] > 0:
                    proto_stats = client.collect_prototypes()
                    try:
                        self._plot_latent_with_prototypes(
                            latents,
                            lat_labels,
                            proto_stats,
                            client_dir,
                            epoch,
                            client_id=idx,
                            method="tsne",
                            max_points=2000,
                            random_state=getattr(self.args, "assign_seed", 0),
                        )
                    except Exception:
                        pass

                    # baseline t-SNE and non-iid-dir 3-color plot
                    try:
                        plot_latent_embedding(latents, lat_labels, client_dir, epoch, client_id=idx, method="tsne")
                    except Exception:
                        pass

                    try:
                        if getattr(self.args, "experiment_type", None) == "non_iid_dir":
                            seen_sets = getattr(self.args, "last_partition_meta", {}) or {}
                            seen_list = seen_sets.get("seen_sets", []) if isinstance(seen_sets, dict) else []
                            seen_for_client = []
                            try:
                                if isinstance(seen_list, list) and idx < len(seen_list):
                                    seen_for_client = list(map(int, seen_list[idx]))
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
                                client_id=idx,
                                method="tsne",
                                max_points=2000,
                                random_state=getattr(self.args, "assign_seed", 0),
                            )
                            if out:
                                self.args.logger.info(f"Saved non-iid-dir latent viz for client {idx} -> {out}")
                    except Exception:
                        pass

                    # Histogram of first latent component
                    try:
                        if getattr(self.args, "experiment_type", None) == "non_iid_dir":
                            hist_p = plot_latent_first_component_hist_from_latents(latents, client_dir, epoch, client_id=idx)
                            if hist_p:
                                self.args.logger.info(f"Saved client {idx} latent-dim0 histogram to {hist_p}")
                            try:
                                first_comp = np.asarray(latents)[:, 0].tolist()
                                global_latent_firstcomp.extend(first_comp)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception as e:
                self.args.logger.warning(f"Failed to collect/plot latent embedding for client {idx}: {e}")

            # accumulate RE/labels for global analytics
            global_re.extend(client_re_list)
            global_labels.extend(client_raw_labels)
            global_client_idxs.extend([idx] * len(client_raw_labels))
            per_client_data[idx] = {
                "re": list(map(float, client_re_list)),
                "labels": list(map(int, client_raw_labels)),
            }

            # predictions-based seen/unseen metrics per client
            if getattr(self.args, "experiment_type", None) == "non_iid_dir":
                try:
                    preds, true_labels_raw = collect_predictions_and_labels_from_client(client)

                    pm = getattr(self.args, "last_partition_meta", None) or {}
                    seen_list = pm.get("seen_sets", []) if isinstance(pm, dict) else []
                    seen_for_client = set()
                    if isinstance(seen_list, list) and idx < len(seen_list):
                        seen_for_client = set(map(int, seen_list[idx]))

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
                            "Client": f"Client {idx}",
                            "Recall_seen": recall_seen * 100 if not np.isnan(recall_seen) else np.nan,
                            "Recall_unseen": recall_unseen * 100 if not np.isnan(recall_unseen) else np.nan,
                            "Recall_normal": recall_normal * 100 if not np.isnan(recall_normal) else np.nan,
                            "Precision_attack": precision_attack * 100 if not np.isnan(precision_attack) else np.nan,
                            "Precision_normal": precision_normal * 100 if not np.isnan(precision_normal) else np.nan,
                        }
                    )
                except Exception as e:
                    self.args.logger.warning(f"Failed detailed metrics for client {idx}: {e}")

            # standard metrics
            acc, precision, recall, f1, roc = client.test_with_logits()

            client.update_nn_parameters(recent_weight_model)

            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            roc_list.append(roc)

            acc_by_attack = client.test_by_attack_type_full()
            acc_class_list.append(acc_by_attack)

            if f1 > best_f1:
                best_f1 = f1
                best_client_idx = idx

            self.args.logger.info(
                f"Client {idx}: Acc {acc:.4f}, Precision {precision:.4f}, Recall {recall:.4f}, F1 {f1:.4f}, ROC {roc:.4f}"
            )

            existing_df = self.args.get_test_log_df()
            new_row = pd.DataFrame(
                [
                    {
                        "epoch": epoch,
                        "client_id": idx,
                        "is_mal": idx in poisoned_workers,
                        "auc": roc,
                        "accuracy": acc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                ]
            )

            if existing_df is None or (isinstance(existing_df, pd.DataFrame) and existing_df.empty):
                self.args.set_test_log_df(new_row)
            else:
                self.args.set_test_log_df(pd.concat([existing_df, new_row], ignore_index=True))

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
                        z_list, _ = collect_z_norms_and_labels_from_client(client)
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

            try:
                pm = getattr(self.args, "last_partition_meta", None) or {}
                seen_sets = pm.get("seen_sets", []) if isinstance(pm, dict) else []

                # per-client metrics (AUC + classification)
                for cidx, pdata in per_client_data.items():
                    try:
                        client_dir = os.path.join(out_dir, f"client_{cidx}")
                        os.makedirs(client_dir, exist_ok=True)
                        client_aucs = compute_auc_per_attack_from_flat(pdata["labels"], pdata["re"])
                        client_aucs_json = {str(int(k)): (None if v is None else float(v)) for k, v in client_aucs.items()}
                        # classification metrics from test_by_attack_type_full
                        try:
                            classif = clients[cidx].test_by_attack_type_full(verbose=False)
                        except Exception:
                            classif = {}

                        seen_set = []
                        try:
                            if isinstance(seen_sets, list) and cidx < len(seen_sets):
                                seen_set = list(map(int, seen_sets[cidx]))
                        except Exception:
                            seen_set = []

                        out_payload = {
                            "client_id": int(cidx),
                            "seen_set": seen_set,
                            "per_attack_auc": client_aucs_json,
                            "per_attack_classification": classif,
                        }
                        client_metrics_path = os.path.join(client_dir, f"epoch{epoch}_client{cidx}_per_attack_metrics.json")
                        with open(client_metrics_path, "w") as _jf:
                            json.dump(out_payload, _jf, indent=2)
                    except Exception as e:
                        self.args.logger.warning(f"Failed to compute/save per-client metrics for client {cidx}: {e}")

                # global seen/unseen grouping for attack_label vs benign
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

                try:
                    if detailed_metrics_rows:
                        detailed_df = pd.DataFrame(detailed_metrics_rows)
                        detail_path = os.path.join(out_dir, f"epoch{epoch}_client_seen_unseen_metrics.csv")
                        detailed_df.to_csv(detail_path, index=False)
                        self.args.logger.info(f"Saved per-client seen/unseen metrics to {detail_path}")
                except Exception as e:
                    self.args.logger.warning(f"Failed to write per-client seen/unseen metrics: {e}")
            except Exception as e:
                self.args.logger.warning(f"Failed to compute/save non_iid_dir specific metrics: {e}")

        return acc_list, precision_list, recall_list, f1_list, roc_list, acc_class_list, best_client_idx

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
