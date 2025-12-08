import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

class ClientPTLAE:
    def __init__(self, args, client_idx, train_data_loader, val_data_loader, test_data_loader):
        self.args = args
        self.client_idx = client_idx
        self.model_type = self.args.model_type
        self.is_training = True
        # bookkeeping attributes expected by server logic
        self.best_loss = 1e9
        self.best_epoch = -1
        self.threshold_re = (1e9, 0.0)
        self.threshold_z = (1e9, 0.0)
        self.best_weight_model = None

        # recent metrics (used for logs)
        self.recent_re = 0.0
        self.recent_latent_z = 0.0
        self.recent_train_loss = 0.0
        self.recent_val_loss = 0.0
        self.recent_threshold_re = (1e9, 0.0)
        self.recent_threshold_z = (1e9, 0.0)

        self.device = self.initialize_device()
        # Use AE net architecture for PTLAE
        from function.nets.ae import AE
        self.net = AE(self.args.dimension)
        self.net.to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.loss_function = self.args.loss_function()

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        # prototypes (server will set before training rounds)
        self.prototype_z0 = None
        self.prototype_z1 = None

    def initialize_device(self):
        if torch.cuda.is_available() and self.args.cuda:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def set_net(self, net):
        self.net = net
        self.net.to(self.device)

    def load_default_model(self):
        """Load default model from default_model folder like other clients."""
        model_class = self.args.get_net(self.model_type)
        default_model_path = os.path.join(self.args.default_model_folder_path, model_class.__name__ + ".model")
        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        model_class = self.args.get_net(self.model_type)
        model = model_class(self.args.dimension)
        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except Exception:
                self.args.logger.warning("Couldn't load model; mapping to CPU")
                model.load_state_dict(torch.load(model_file_path, map_location=torch.device("cpu")))
        else:
            self.args.logger.warning(f"Could not find model: {model_file_path}")
        return model

    def set_recent_metric(self, recent_re, recent_latent_z, recent_train_loss, recent_val_loss, recent_threshold_re, recent_threshold_z):
        self.recent_re = recent_re
        self.recent_latent_z = recent_latent_z
        self.recent_train_loss = recent_train_loss
        self.recent_val_loss = recent_val_loss
        self.recent_threshold_re = recent_threshold_re
        self.recent_threshold_z = recent_threshold_z

    def visualize(self, epoch):
        # similar to ClientDualLossAE.visualize: scatter latent first two dims colored by label
        self.net.eval()
        benign_X = []
        benign_y = []
        mal_X = []
        mal_y = []
        with torch.no_grad():
            for inputs, labels in self.test_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                encode, _ = self.net(inputs)
                list_label = labels.tolist()
                list_encode = encode.tolist()
                for i in range(len(list_label)):
                    if list_label[i] == 1:
                        mal_X.append(list_encode[i][0])
                        mal_y.append(list_encode[i][1])
                    else:
                        benign_X.append(list_encode[i][0])
                        benign_y.append(list_encode[i][1])
        plt.figure()
        plt.scatter(mal_X, mal_y, c="red", marker="x")
        plt.scatter(benign_X, benign_y, c="blue", marker="o")
        out_dir = os.path.join("visual", self.model_type)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"epoch_{epoch}_{self.model_type}_client{self.client_idx}.pdf"))
        plt.close()

    def visualize_z(self, epoch):
        # t-SNE over train latents
        self.net.eval()
        latent_z_list = []
        with torch.no_grad():
            for input, _ in self.train_data_loader:
                input = input.to(self.device)
                encode, _ = self.net(input)
                latent_z_list.append(encode.cpu().numpy())
        if len(latent_z_list) == 0:
            return
        latent_z_array = np.concatenate(latent_z_list, axis=0)
        if len(latent_z_array) > 200:
            idxs = np.random.choice(len(latent_z_array), size=200, replace=False)
            latent_z_array = latent_z_array[idxs]
        tsne = TSNE(n_components=2, random_state=42)
        z_embedded = tsne.fit_transform(latent_z_array)
        plt.figure(figsize=(8, 8))
        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c="blue", label=f"Client {self.client_idx}")
        plt.title(f"t-SNE of Latent z for Client {self.client_idx} at Epoch {epoch}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid()
        out_dir = os.path.join("visual", self.model_type)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"epoch_{epoch}_{self.model_type}_client{self.client_idx}_tsne.png"))
        plt.close()

    def set_prototypes(self, z0, z1):
        """Set prototypes (torch tensors on CPU or same device)."""
        # prototypes expected as numpy arrays or torch tensors
        # set each prototype independently so a missing z0/z1 doesn't clear the other
        if z0 is None:
            self.prototype_z0 = None
        else:
            if not isinstance(z0, torch.Tensor):
                z0 = torch.tensor(z0, dtype=torch.float32)
            self.prototype_z0 = z0.to(self.device)

        if z1 is None:
            self.prototype_z1 = None
        else:
            if not isinstance(z1, torch.Tensor):
                z1 = torch.tensor(z1, dtype=torch.float32)
            self.prototype_z1 = z1.to(self.device)

    def get_nn_parameters(self):
        return self.net.state_dict()

    def update_nn_parameters(self, new_params):
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def prototype_triplet_loss(self, z_i, y_i, z0, z1, margin=1.0, distance='euclid'):
        """z_i: (B, dim), y_i: (B,), z0,z1: (dim,)"""
        if z0 is None or z1 is None:
            # no prototype available yet
            return torch.tensor(0.0, device=z_i.device)

        if distance == 'euclid':
            d = lambda a, b: torch.norm(a - b, dim=1)
        elif distance == 'cosine':
            d = lambda a, b: 1 - F.cosine_similarity(a, b)
        else:
            raise ValueError("distance must be 'euclid' or 'cosine'")

        # broadcast prototypes to batch
        positive = torch.where(y_i.unsqueeze(1) == 1, z1.unsqueeze(0).expand(z_i.size(0), -1), z0.unsqueeze(0).expand(z_i.size(0), -1))
        negative = torch.where(y_i.unsqueeze(1) == 1, z0.unsqueeze(0).expand(z_i.size(0), -1), z1.unsqueeze(0).expand(z_i.size(0), -1))

        d_pos = d(z_i, positive)
        d_neg = d(z_i, negative)

        loss = F.relu(d_pos - d_neg + margin)
        return loss.mean()

    def train(self, epoch):
        """Train one epoch and collect per-class latent sums and counts to send to server.
        Returns: (avg_loss_float, local_proto_stats dict)
        local_proto_stats: {label: (sum_vector_cpu, count)}
        """
        self.net.train()
        latent_z_dict = defaultdict(list)
        total_loss = 0.0
        it = 0

        total_re = 0.0
        total_z_norm = 0.0
        for input, label in self.train_data_loader:
            input, label = input.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            encode, decode = self.net(input)
            re_loss = self.loss_function(decode, input)

            # prototype triplet loss
            ptl = self.prototype_triplet_loss(encode, label, self.prototype_z0, self.prototype_z1, margin=1.0, distance='euclid')

            lambda_ptl = getattr(self.args, 'ptl_lambda', 1.0)
            loss = re_loss + lambda_ptl * ptl
            loss.backward()
            self.optimizer.step()

            re_val = float(re_loss.item())
            total_re += re_val
            total_loss += loss.item()
            # accumulate latent norm
            z_norms = torch.norm(encode.view(encode.size(0), -1), p=2, dim=1)
            total_z_norm += float(z_norms.mean().item())
            it += 1

            # accumulate latents per-sample for prototype stats (CPU)
            encode_cpu = encode.detach().cpu()
            labels_cpu = label.detach().cpu().tolist()
            for idx, lab in enumerate(labels_cpu):
                latent_z_dict[int(lab)].append(encode_cpu[idx])

        avg_loss = total_loss / it if it > 0 else 0.0

        # compute sums and counts per class
        local_proto_stats = {}
        for lab, latents in latent_z_dict.items():
            if len(latents) == 0:
                continue
            stacked = torch.stack(latents, dim=0)  # (N, dim)
            sum_vec = stacked.sum(dim=0).numpy()
            count = stacked.size(0)
            local_proto_stats[int(lab)] = (sum_vec, int(count))
        # compute per-client mean prototypes (z0, z1) for logging
        z0 = None
        z1 = None
        if 0 in local_proto_stats:
            s0, c0 = local_proto_stats[0]
            z0 = (s0 / max(1, c0)).astype(float)
        if 1 in local_proto_stats:
            s1, c1 = local_proto_stats[1]
            z1 = (s1 / max(1, c1)).astype(float)
        # store as numpy arrays for later access
        if z0 is not None:
            self.prototype_z0 = z0
        if z1 is not None:
            self.prototype_z1 = z1

        # compute and store recent per-epoch metrics used by server logging
        avg_re = total_re / it if it > 0 else 0.0
        avg_z = total_z_norm / it if it > 0 else 0.0
        self.recent_re = avg_re
        self.recent_latent_z = avg_z
        self.recent_train_loss = avg_loss

        return avg_loss, local_proto_stats

    def validate(self, epoch):
        # similar to ClientDualLossAE validate: compute val loss and threshold based on benign
        self.net.eval()
        with torch.no_grad():
            re_loss_label_0 = []
            list_loss = []
            for input, label in self.val_data_loader:
                input, label = input.to(self.device), label.to(self.device)
                _, decode = self.net(input)
                re_loss = self.loss_function(decode, input).mean().item()
                list_loss.append(re_loss)
                if label.item() == 0:
                    re_loss_label_0.append(re_loss)

            avg_loss = float(np.mean(list_loss)) if list_loss else 0.0
            threshold_re = ((np.mean(re_loss_label_0), np.std(re_loss_label_0)) if re_loss_label_0 else (0,0))
            threshold_z = (0,0)
            return avg_loss, threshold_re, threshold_z

    def test(self, is_check=False):
        # perform a threshold sweep to match other client APIs
        acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_list = []
        mean_re, std_re = self.threshold_re if hasattr(self, "threshold_re") and isinstance(self.threshold_re, tuple) else (0.0, 0.0)

        multipliers = np.arange(0.0, 5.1, 0.2)

        # If using prototype-only decision, prototype distances don't depend on reconstruction-error thresholds.
        # Compute metrics once and repeat across multipliers so server aggregation logic stays compatible.
        decision_mode = getattr(self.args, 'ptl_decision_mode', 're')
        if decision_mode == 'proto':
            # choose a fallback threshold_re for the (rare) case where proto path cannot compute distances
            threshold_re = mean_re
            # When running proto-mode, produce verbose output (confusion matrix/class report) unless explicitly in check mode
            is_verbose = (not is_check)
            acc, precision, recall, f1, roc = self.test_with_thresholds(threshold_re, is_verbose)
            # repeat results for each multiplier so caller (server) can iterate as before
            acc_list = [acc for _ in multipliers]
            precision_list = [precision for _ in multipliers]
            recall_list = [recall for _ in multipliers]
            f1_list = [f1 for _ in multipliers]
            roc_list = [roc for _ in multipliers]
            return acc_list, precision_list, recall_list, f1_list, roc_list

        # default: RE-based sweep
        for multiplier in multipliers:
            threshold_re = mean_re + multiplier * std_re
            is_verbose = (not is_check and abs(multiplier - self.args.threshold_multiplier) < 1e-4)
            acc, precision, recall, f1, roc = self.test_with_thresholds(threshold_re, is_verbose)
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            roc_list.append(roc)

        return acc_list, precision_list, recall_list, f1_list, roc_list

    def set_best_ckpt(self, best_loss, best_epoch, threshold_re, threshold_z, best_weight_model):
        self.best_loss = best_loss
        self.best_epoch = best_epoch
        self.threshold_re = threshold_re
        # store the client's best prototypes at the time of best checkpoint
        self.best_proto_z0 = getattr(self, 'prototype_z0', None)
        self.best_proto_z1 = getattr(self, 'prototype_z1', None)
        self.best_weight_model = copy.deepcopy(best_weight_model)

    def set_training_status(self, status):
        self.is_training = status

    def test_with_thresholds(self, threshold_re, verbose=False, show_samples=False, sample_n=5):
        """
        Test using a fixed threshold. Returns acc, precision, recall, f1, roc.
        """
        self.net.eval()
        with torch.no_grad():
            list_re = []
            labels = []
            sample_list = []

            # Decide mode: 're' (default), 'proto' (proto-only), or 'combined' (not used here)
            decision_mode = getattr(self.args, 'ptl_decision_mode', 're')
            encodes_for_proto = []
            for input, label in self.test_data_loader:
                input, label = input.to(self.device), label.to(self.device)
                encode, decode = self.net(input)
                per_sample_re = ((decode - input) ** 2).mean(dim=1).cpu().numpy().tolist()
                list_re.extend(per_sample_re)
                labels += label.cpu().tolist()

                if decision_mode == 'proto':
                    # collect encodings for prototype distance decision
                    encodes_for_proto.append(encode.detach().cpu())

                if show_samples and len(sample_list) < sample_n:
                    for lab_val, re_val in zip(label.cpu().tolist(), per_sample_re):
                        if len(sample_list) >= sample_n:
                            break
                        sample_list.append((int(lab_val), float(re_val)))

            # Ensure labels are binary (benign=0, malicious=1) when computing binary metrics.
            # Some datasets may carry multi-class attack ids; convert them to binary here.
            labels_bin = [int(l != 0) for l in labels]
            # If by_attack_type is set, original code expected labels to be binary already; keep for compatibility
            if getattr(self.args, 'by_attack_type', False):
                labels = labels_bin

            # If prototype-only decision is requested and prototypes exist, use proto distances
            if decision_mode == 'proto' and self.prototype_z0 is not None and self.prototype_z1 is not None:
                # concatenate encodings
                if len(encodes_for_proto) == 0:
                    predictions = [1 if r > threshold_re else 0 for r in list_re]
                else:
                    enc_cat = torch.cat(encodes_for_proto, dim=0)
                    # Log prototype info (shape, norm, head elements) for debugging before computing distances
                    def _vec_info(v):
                        if isinstance(v, torch.Tensor):
                            arr = v.detach().cpu().numpy()
                        else:
                            arr = np.array(v)
                        return {"shape": tuple(arr.shape), "norm": float(np.linalg.norm(arr)), "head": arr.reshape(-1)[:6].tolist()}

                    z0_info = _vec_info(self.prototype_z0)
                    z1_info = _vec_info(self.prototype_z1)
                    self.args.logger.info(f"[Client {self.client_idx}] Using prototypes for proto decision: proto_z0={z0_info}, proto_z1={z1_info}")

                    # move prototypes to CPU tensors
                    z0 = self.prototype_z0.cpu() if isinstance(self.prototype_z0, torch.Tensor) else torch.tensor(self.prototype_z0)
                    z1 = self.prototype_z1.cpu() if isinstance(self.prototype_z1, torch.Tensor) else torch.tensor(self.prototype_z1)
                    # choose distance metric
                    distance = getattr(self.args, 'ptl_distance', 'euclid')
                    if distance == 'euclid':
                        d_pos = torch.norm(enc_cat - z1.unsqueeze(0).expand(enc_cat.size(0), -1), dim=1).numpy().tolist()
                        d_neg = torch.norm(enc_cat - z0.unsqueeze(0).expand(enc_cat.size(0), -1), dim=1).numpy().tolist()
                    elif distance == 'cosine':
                        import torch.nn.functional as F
                        d_pos = (1 - F.cosine_similarity(enc_cat, z1.unsqueeze(0).expand(enc_cat.size(0), -1))).numpy().tolist()
                        d_neg = (1 - F.cosine_similarity(enc_cat, z0.unsqueeze(0).expand(enc_cat.size(0), -1))).numpy().tolist()
                    else:
                        raise ValueError("Unsupported ptl_distance: {}".format(distance))

                    # prediction: closer to malicious prototype -> predict malicious (1)
                    # optionally apply margin
                    margin = getattr(self.args, 'ptl_margin', 0.0)
                    predictions = [1 if (dp + margin) < dn else 0 for dp, dn in zip(d_pos, d_neg)]
                    # for ROC scoring use proto_score = d_neg - d_pos (higher -> more likely malicious)
                    proto_scores = [dn - dp for dp, dn in zip(d_pos, d_neg)]
            else:
                predictions = [1 if r > threshold_re else 0 for r in list_re]

            # Use binarized labels for binary classification metrics
            y_true_bin = labels_bin
            acc = accuracy_score(y_true_bin, predictions)
            precision = precision_score(y_true_bin, predictions, zero_division=0)
            recall = recall_score(y_true_bin, predictions, zero_division=0)
            f1 = f1_score(y_true_bin, predictions, zero_division=0)
            # ROC needs binary labels as well
            if decision_mode == 'proto' and self.prototype_z0 is not None and self.prototype_z1 is not None:
                roc = roc_auc_score(y_true_bin, proto_scores) if len(set(y_true_bin)) > 1 else 0.0
            else:
                roc = roc_auc_score(y_true_bin, list_re) if len(set(y_true_bin)) > 1 else 0.0

            if verbose:
                # show report/confusion on binarized labels
                self.args.logger.debug("Classification Report:\n" + classification_report(y_true_bin, predictions))
                self.args.logger.debug("Confusion Matrix:\n" + str(confusion_matrix(y_true_bin, predictions)))
                self.args.logger.debug("ROC AUC Score: {}".format(roc))

            if show_samples and len(sample_list) > 0:
                sample_with_pred = []
                for lab_val, re_val in sample_list:
                    lab_bin = int(lab_val != 0) if self.args.by_attack_type else int(lab_val)
                    pred = 1 if re_val > threshold_re else 0
                    sample_with_pred.append((lab_bin, float(re_val), int(pred)))
                self.args.logger.info(f"Sample label, RE, pred (first {len(sample_with_pred)}): {sample_with_pred}")

            return acc, precision, recall, f1, roc

    def test_by_attack_type_full(self, threshold_re, threshold_z, verbose=False):
        """
        Compute per-attack metrics using given threshold_re.
        Returns dict attack_id -> metrics.
        """
        self.net.eval()
        with torch.no_grad():
            list_re = []
            bin_labels = []
            raw_labels = []

            for input, label in self.test_data_loader:
                input, label = input.to(self.device), label.to(self.device)
                _, decode = self.net(input)
                per_sample_re = ((decode - input) ** 2).mean(dim=1).cpu().numpy().tolist()
                list_re.extend(per_sample_re)
                for l in label.cpu().tolist():
                    bin_labels.append(int(l != 0))
                    raw_labels.append(int(l))

            predictions = [1 if r > threshold_re else 0 for r in list_re]

            results_by_type = defaultdict(lambda: {"y_true": [], "y_pred": []})
            for y, pred, atk_type in zip(bin_labels, predictions, raw_labels):
                results_by_type[atk_type]["y_true"].append(y)
                results_by_type[atk_type]["y_pred"].append(pred)

            metrics_by_type = {}
            for atk_type, data in results_by_type.items():
                y_true = data["y_true"]
                y_pred = data["y_pred"]
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                acc = accuracy_score(y_true, y_pred)
                metrics_by_type[atk_type] = {
                    "acc": acc,
                    "precision": report.get("1", {}).get("precision", 0.0),
                    "recall": report.get("1", {}).get("recall", 0.0),
                    "f1-score": report.get("1", {}).get("f1-score", 0.0),
                    "support": len(y_true),
                }
                if verbose:
                    print(f"\n=== Attack Type {atk_type} ===")
                    print(f"Accuracy: {acc:.4f}")
                    print(classification_report(y_true, y_pred, zero_division=0))
                    print("Confusion Matrix:")
                    print(confusion_matrix(y_true, y_pred))

            return metrics_by_type
