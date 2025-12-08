import torch
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import os
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict


class ClientDualLossAE:
    def __init__(
        self, args, client_idx, train_data_loader, val_data_loader, test_data_loader
    ):
        """
        :param args: experiment arguments
        :param client_idx: Client index
        :param train_data_loader: Training data loader
        :param val_data_loader: Val data loader
        :param test_data_loader: Test data loader
        """
        self.args = args
        self.client_idx = client_idx
        self.model_type = self.args.model_type
        self.is_training = True

        self.best_loss = 1e9
        self.best_epoch = -1
        self.threshold_re = 1e9
        self.threshold_z = 1e9
        self.best_weight_model = None

        # use for log, not for train/test
        self.recent_re = 0.0
        self.recent_latent_z = 0.0
        self.recent_train_loss = 0.0
        self.recent_val_loss = 0.0
        self.recent_threshold_re = (1e9, 0)
        self.recent_threshold_z = (1e9, 0)

        self.device = self.initialize_device()
        self.set_net(self.load_default_model())

        self.loss_function = self.args.loss_function()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.learning_rate)

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        if torch.cuda.is_available() and self.args.cuda:
            # print("Device: GPU")
            return torch.device("cuda:0")
        else:
            # print("Device: CPU")
            return torch.device("cpu")

    def set_net(self, net):
        """
        Set the client's NN.
        """
        self.net = net
        self.net.to(self.device)

    def set_best_ckpt(
        self, best_loss, best_epoch, threshold_re, threshold_z, best_weight_model
    ):
        """
        Set the best training point.
        """
        self.best_loss = best_loss
        self.best_epoch = best_epoch
        self.threshold_re = threshold_re
        self.threshold_z = threshold_z
        self.best_weight_model = copy.deepcopy(best_weight_model)

    def set_training_status(self, training_status):
        """
        Set the training status.
        """
        self.is_training = training_status

    def set_recent_metric(
        self,
        recent_re,
        recent_latent_z,
        recent_train_loss,
        recent_val_loss,
        recent_threshold_re,
        recent_threshold_z,
    ):
        """
        Set the recent metrics for logging.
        """
        self.recent_re = recent_re
        self.recent_latent_z = recent_latent_z
        self.recent_train_loss = recent_train_loss
        self.recent_val_loss = recent_val_loss
        self.recent_threshold_re = recent_threshold_re
        self.recent_threshold_z = recent_threshold_z

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net(self.model_type)
        default_model_path = os.path.join(
            self.args.default_model_folder_path, model_class.__name__ + ".model"
        )

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.
        """
        model_class = self.args.get_net(self.model_type)
        model = model_class(self.args.dimension)

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.logger.warning(
                    "Couldn't load model. Attempting to map CUDA tensors to CPU to solve error."
                )

                model.load_state_dict(
                    torch.load(model_file_path, map_location=torch.device("cpu"))
                )
        else:
            self.args.logger.warning("Could not find model: {}".format(model_file_path))

        return model

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def calculate_loss(self, input, label):
        """
        Calculate the loss value.

        :param input: Training data sample
        """
        if self.model_type == "DualLossAE" or self.model_type == "SupAE":
            return self.net.calculate_loss(input, label)

    def train(self, epoch):
        self.net.train()
        for input, label in self.train_data_loader:
            input, label = input.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            loss = self.calculate_loss(input, label)
            loss.backward()
            self.optimizer.step()
        return loss

    def validate(self, epoch):
        self.net.eval()
        with torch.no_grad():
            re_loss_label_0 = []
            list_loss = []
            for input, label in self.val_data_loader:
                input, label = input.to(self.device), label.to(self.device)

                loss = self.calculate_loss(input, label)
                _, decode = self.net(input)
                re_loss = self.loss_function(decode, input).mean().item()

                list_loss.append(loss.cpu().numpy().tolist())

                if label.item() == 0:
                    re_loss_label_0.append(re_loss)

            avg_loss = np.mean(list_loss)
            threshold_re = (
                (np.mean(re_loss_label_0), np.std(re_loss_label_0))
                if re_loss_label_0
                else (0, 0)
            )
            threshold_z = (0, 0)

            return avg_loss, threshold_re, threshold_z

    def test(self, is_check=False):
        """
        Test với nhiều giá trị threshold_multiplier (0.0 → 2.0, bước 0.2).
        """
        acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_list = []

        mean_re, std_re = self.threshold_re

        for multiplier in np.arange(0.0, 5.1, 0.2):
            threshold_re = mean_re + multiplier * std_re

            # print(f"[Client {self.client_idx}] Testing with multiplier {multiplier:.1f} - threshold_re: {threshold_re:.4f}")
            is_verbose = (
                not is_check and abs(multiplier - self.args.threshold_multiplier) < 1e-4
            )

            acc, precision, recall, f1, roc = self.test_with_thresholds(
                threshold_re, is_verbose, is_verbose, 50
            )
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            roc_list.append(roc)

        return acc_list, precision_list, recall_list, f1_list, roc_list

    def test_with_thresholds(self, threshold_re, verbose=False, show_samples=False, sample_n=50):
        """
        Test with a fixed threshold. Optionally log a few (label, re) samples.

        :param threshold_re: threshold on reconstruction error
        :param verbose: verbose logging for metrics
        :param show_samples: if True, log the first `sample_n` (label, re) pairs
        :param sample_n: number of samples to log when show_samples=True
        """
        self.net.eval()
        with torch.no_grad():
            list_re = []
            labels = []
            sample_list = []

            for input, label in self.test_data_loader:
                input, label = input.to(self.device), label.to(self.device)
                _, decode = self.net(input)

                # compute per-sample reconstruction error (MSE per sample)
                per_sample_re = ((decode - input) ** 2).mean(dim=1).cpu().numpy().tolist()

                list_re.extend(per_sample_re)
                labels += label.cpu().tolist()

                # collect some samples for quick inspection
                if show_samples and len(sample_list) < sample_n:
                    # append per-sample pairs until we reach sample_n
                    for lab_val, re_val in zip(label.cpu().tolist(), per_sample_re):
                        if len(sample_list) >= sample_n:
                            break
                        sample_list.append((int(lab_val), float(re_val)))

            # nếu đang chạy thí nghiệm theo loại attack type thì ép hết label tấn công về 1
            if self.args.by_attack_type:
                labels = [int(l != 0) for l in labels]

            predictions = [1 - int(r <= threshold_re) for r in zip(list_re)]

            acc = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions)
            recall = recall_score(labels, predictions)
            f1 = f1_score(labels, predictions)
            roc = roc_auc_score(labels, predictions)

            # Tính confusion matrix và FPR
            confusion_mat = confusion_matrix(labels, predictions)
            tn, fp, fn, tp = confusion_mat.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            if verbose:
                confusion_mat = confusion_matrix(labels, predictions)
                self.args.logger.debug(
                    "Classification Report:\n"
                    + classification_report(labels, predictions)
                )
                self.args.logger.debug("Confusion Matrix:\n" + str(confusion_mat))
                self.args.logger.debug("ROC AUC Score: {}".format(roc))
                self.args.logger.debug("False Positive Rate (FPR): {:.4f}".format(fpr))

            if show_samples and len(sample_list) > 0:
                # compute prediction for each sampled pair and log (label, re, pred)
                sample_list_with_pred = []
                for lab_val, re_val in sample_list:
                    # apply by_attack_type conversion to label if needed
                    lab_bin = int(lab_val != 0) if self.args.by_attack_type else int(lab_val)
                    pred = 1 if re_val > threshold_re else 0
                    sample_list_with_pred.append((lab_bin, float(re_val), int(pred)))

                self.args.logger.info(f"Sample label, RE, pred (first {len(sample_list_with_pred)}): {sample_list_with_pred}")
               
            return acc, precision, recall, f1, roc

    def test_by_attack_type_full(self, threshold_re, threshold_z, verbose=False):
        self.net.eval()
        with torch.no_grad():
            list_re = []
            bin_labels = []  # binary: 0 (normal), 1 (attack)
            raw_labels = []  # gốc: 0 (normal), 1, 2, ...

            for input, label in self.test_data_loader:
                input, label = input.to(self.device), label.to(self.device)
                _, decode = self.net(input)
                re = self.loss_function(decode, input).mean().item()

                list_re.append(re)

                binary_label = int(label.item() != 0)
                bin_labels.append(binary_label)
                raw_labels.append(int(label.item()))

            # Dự đoán dựa vào threshold_re
            predictions = [1 if re > threshold_re else 0 for re in list_re]

            # Gom kết quả theo loại attack type
            results_by_type = defaultdict(lambda: {"y_true": [], "y_pred": []})
            for y, pred, atk_type in zip(bin_labels, predictions, raw_labels):
                results_by_type[atk_type]["y_true"].append(y)
                results_by_type[atk_type]["y_pred"].append(pred)

            # Tính metric cho từng loại attack
            metrics_by_type = {}
            for atk_type, data in results_by_type.items():
                y_true = data["y_true"]
                y_pred = data["y_pred"]
                report = classification_report(
                    y_true, y_pred, output_dict=True, zero_division=0
                )
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

    def visualize(self, epoch):
        self.net.eval()

        with torch.no_grad():
            benign_X = []
            benign_y = []
            mal_X = []
            mal_y = []

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
            plt.savefig(
                "./visual/{}/epoch_{}_{}.pdf".format(
                    self.model_type, epoch, self.model_type
                )
            )
            plt.close()

    def visualize_z(self, epoch):
        """
        Visualize latent z using t-SNE and save to a PDF.

        :param epoch: Current epoch
        :param pdf_writer: PdfPages writer to save the plot
        """

        self.net.eval()  # Set model to evaluation mode
        latent_z_list = []

        with torch.no_grad():
            for input, _ in self.train_data_loader:
                input = input.to(self.device)

                # Get latent representation
                encode, _ = self.net(input)
                latent_z_list.append(encode.cpu().numpy())

        # Concatenate all latent representations
        latent_z_array = np.concatenate(latent_z_list, axis=0)

        # Randomly select 200 samples if there are more than 200
        if len(latent_z_array) > 200:
            random_indices = np.random.choice(
                len(latent_z_array), size=200, replace=False
            )
            latent_z_array = latent_z_array[random_indices]

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        z_embedded = tsne.fit_transform(latent_z_array)

        # Plot t-SNE results
        plt.figure(figsize=(8, 8))
        plt.scatter(
            z_embedded[:, 0],
            z_embedded[:, 1],
            c="blue",
            label=f"Client {self.client_idx}",
        )
        plt.title(f"t-SNE of Latent z for Client {self.client_idx} at Epoch {epoch}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid()

        # Save to PDF
    out_dir = os.path.join("visual", self.model_type)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}_{self.model_type}_client{self.client_idx}_tsne.png"))
        plt.close()

    def visualize_validate(
        self, epoch, threshold_re, threshold_z, auc, list_re, list_latent_z, labels
    ):
        """
        Vẽ biểu đồ phân tán (scatter plot) với:
        - X: Latent Z
        - Y: Reconstruction Error (Re)
        - Phân biệt bằng màu dựa vào label (0: xanh, 1: đỏ)
        - Đường thẳng thể hiện threshold_re và threshold_z
        """
        # os.makedirs("visuals_cic_ids ", exist_ok=True)  # Đảm bảo thư mục tồn tại

        # **Chỉ lấy 100 mẫu dữ liệu đầu tiên**
        num_samples = min(150, len(list_re))
        list_re = np.array(list_re[:num_samples])
        list_latent_z = np.array(list_latent_z[:num_samples])
        labels = np.array(labels[:num_samples])

        # **Tên file**
        file_name = f"{self.args.dataset}_{self.args.model_type}_{self.args.aggregation_type}_multi{self.args.threshold_multiplier}_poison{self.args.num_poisoned_workers}_epoch{epoch}_client{self.client_idx}.png"
        file_path = os.path.join("visual_unsw_test_thres", file_name)

        # **Vẽ biểu đồ**
        plt.figure(figsize=(8, 6))
        plt.title(
            f"Client {self.client_idx} | Epoch {epoch} | AUC: {auc:.6f}\nThresh_Re: {threshold_re:.6f} | Thresh_Z: {threshold_z:.6f}"
        )
        plt.xlabel("Latent Z")
        plt.ylabel("Reconstruction Error")

        # **Vẽ điểm dữ liệu**
        plt.scatter(
            list_latent_z[labels == 0],
            list_re[labels == 0],
            c="blue",
            s=12,
            label="Normal (0)",
            alpha=0.4,
        )
        plt.scatter(
            list_latent_z[labels == 1],
            list_re[labels == 1],
            c="red",
            s=12,
            label="Anomalous (1)",
            alpha=0.4,
        )

        # **Vẽ threshold lines**
        plt.axhline(y=threshold_re, color="black", linestyle="--", label="Threshold Re")
        plt.axvline(x=threshold_z, color="green", linestyle="--", label="Threshold Z")

        plt.legend()
        plt.grid()
        plt.savefig(file_path, dpi=300)
        plt.close()

        # print(f"Saved validation visualization for Client {self.client_idx}: {file_path}")
