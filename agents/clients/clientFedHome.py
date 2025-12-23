import os
import copy
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix


class ClientFedHome:
    def __init__(self, args, client_idx, train_data_loader, val_data_loader, test_data_loader):
        self.args = args
        self.client_idx = client_idx
        self.model_type = self.args.model_type
        self.is_training = True

        self.best_loss = 1e9
        self.best_epoch = -1
        self.threshold_re = (1e9, 0.0)
        self.threshold_z = (1e9, 0.0)
        self.best_weight_model = None

        self.recent_re = 0.0
        self.recent_latent_z = 0.0
        self.recent_train_loss = 0.0
        self.recent_val_loss = 0.0
        self.recent_threshold_re = (1e9, 0.0)
        self.recent_threshold_z = (1e9, 0.0)

        self.device = self.initialize_device()
        self.set_net(self.load_default_model())
        self.best_weight_model = copy.deepcopy(self.net.state_dict())

            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
            self.ce_loss = nn.CrossEntropyLoss()
            self.re_loss = nn.MSELoss()
            self.lambda_ce = getattr(self.args, "lambda_ce", 1.0)
            self.lambda_re = getattr(self.args, "lambda_re", 1.0)

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def initialize_device(self):
        if torch.cuda.is_available() and self.args.cuda:
            return torch.device("cuda:0")
        return torch.device("cpu")

    def set_net(self, net):
        self.net = net
        self.net.to(self.device)

    def load_default_model(self):
        model_class = self.args.get_net(self.model_type)
        n_classes = getattr(self.args, "num_classes", 2)
        default_model_path = os.path.join(self.args.default_model_folder_path, model_class.__name__ + ".model")
        model = model_class(self.args.dimension, n_classes=n_classes)
        if os.path.exists(default_model_path):
            try:
                model.load_state_dict(torch.load(default_model_path))
            except Exception:
                self.args.logger.warning("Couldn't load model; mapping to CPU")
                model.load_state_dict(torch.load(default_model_path, map_location=torch.device("cpu")))
        else:
            self.args.logger.warning(f"Could not find model: {default_model_path}")
        return model

    def get_nn_parameters(self):
        return self.net.state_dict()

    def update_nn_parameters(self, new_params):
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def set_best_ckpt(self, best_loss, best_epoch, threshold_re, threshold_z, best_weight_model):
        self.best_loss = best_loss
        self.best_epoch = best_epoch
        self.threshold_re = threshold_re
        self.threshold_z = threshold_z
        self.best_weight_model = copy.deepcopy(best_weight_model)

    def set_training_status(self, training_status):
        self.is_training = training_status

    def set_recent_metric(self, recent_re, recent_latent_z, recent_train_loss, recent_val_loss, recent_threshold_re, recent_threshold_z):
        self.recent_re = recent_re
        self.recent_latent_z = recent_latent_z
        self.recent_train_loss = recent_train_loss
        self.recent_val_loss = recent_val_loss
        self.recent_threshold_re = recent_threshold_re
        self.recent_threshold_z = recent_threshold_z

    def calculate_loss(self, input, label_bin, logits=None, recon=None):
        if logits is None or recon is None:
            logits, recon = self.net(input)
        ce = self.ce_loss(logits, label_bin)
        re = self.re_loss(recon, input)
        total = self.lambda_ce * ce + self.lambda_re * re
        return total, ce, re, recon
        return total_loss, ce_loss, re_loss

    def train(self, epoch):
        self.net.train()
        last_total = torch.tensor(0.0, device=self.device)
        for input, label in self.train_data_loader:
            input, label = input.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            label_bin = (label != 0).long()
            logits, recon = self.net(input)
            total_loss, _, re_loss, _ = self.calculate_loss(input, label_bin, logits, recon)
            total_loss.backward()
            self.optimizer.step()
            last_total = total_loss.detach()
            self.recent_re = float(re_loss.detach().cpu().item())
            self.recent_train_loss = float(total_loss.detach().cpu().item())
        return last_total

    def validate(self, epoch):
        self.net.eval()
        list_loss = []
        benign_re = []
        with torch.no_grad():
            for input, label in self.val_data_loader:
                input, label = input.to(self.device), label.to(self.device)
                label_bin = (label != 0).long()
                logits, recon = self.net(input)
                total_loss, _, _, recon = self.calculate_loss(input, label_bin, logits, recon)
                list_loss.append(float(total_loss.item()))

                mse_per_sample = torch.mean((recon - input) ** 2, dim=1)
                benign_mask = label_bin == 0
                if benign_mask.any():
                    benign_re.extend(mse_per_sample[benign_mask].detach().cpu().tolist())

        avg_loss = float(np.mean(list_loss)) if list_loss else 0.0
        threshold_re = ((float(np.mean(benign_re)), float(np.std(benign_re))) if benign_re else (0.0, 0.0))
        threshold_z = (0.0, 0.0)
        return avg_loss, threshold_re, threshold_z

    def test(self, is_check=False):
        acc_list, precision_list, recall_list, f1_list, roc_list = [], [], [], [], []
        acc, precision, recall, f1, roc = self.test_with_logits()
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        roc_list.append(roc)
        return acc_list, precision_list, recall_list, f1_list, roc_list

    def test_with_logits(self, verbose=False):
        self.net.eval()
        labels_bin = []
        preds_bin = []
        prob_attack = []
        with torch.no_grad():
            for input, label in self.test_data_loader:
                input, label = input.to(self.device), label.to(self.device)
                logits, recon = self.net(input)
                scores = torch.softmax(logits, dim=1)
                    prob1 = scores[:, 1] if scores.shape[1] > 1 else scores[:, 0]
                pred = torch.argmax(scores, dim=1)
                labels_bin += (label != 0).long().cpu().tolist()
                preds_bin += pred.cpu().tolist()
                prob_attack += prob1.cpu().tolist()
                self.recent_re = float(torch.mean((recon - input) ** 2).item())
        if len(set(labels_bin)) <= 1:
            roc = 0.0
        else:
            roc = float(roc_auc_score(labels_bin, prob_attack))
        acc = accuracy_score(labels_bin, preds_bin)
        precision = precision_score(labels_bin, preds_bin, zero_division=0)
        recall = recall_score(labels_bin, preds_bin, zero_division=0)
        f1 = f1_score(labels_bin, preds_bin, zero_division=0)

        if verbose:
            self.args.logger.debug(
                "Classification Report:\n" + classification_report(labels_bin, preds_bin, zero_division=0)
            )
            self.args.logger.debug("Confusion Matrix:\n" + str(confusion_matrix(labels_bin, preds_bin, labels=[0, 1])))
            self.args.logger.debug(f"ROC AUC Score: {roc}")

        return acc, precision, recall, f1, roc

    def test_by_attack_type_full(self, threshold_re_unused, threshold_z_unused, verbose=False):
        self.net.eval()
        labels_raw = []
        preds = []
        with torch.no_grad():
            for input, label in self.test_data_loader:
                input, label = input.to(self.device), label.to(self.device)
                logits, _ = self.net(input)
                pred_bin = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                labels_raw.extend(label.cpu().tolist())
                preds.extend(pred_bin.cpu().tolist())

        labels_arr = np.array(labels_raw, dtype=int)
        preds_arr = np.array(preds, dtype=int)
        metrics_by_type = {}
        for atk_type in sorted(set(labels_arr.tolist())):
            if atk_type == 0:
                continue
            mask = np.isin(labels_arr, [0, atk_type])
            if mask.sum() == 0:
                continue
            y_true_bin = (labels_arr[mask] == atk_type).astype(int)
            y_pred_bin = (preds_arr[mask] != 0).astype(int)
            acc = accuracy_score(y_true_bin, y_pred_bin)
            report = classification_report(y_true_bin, y_pred_bin, output_dict=True, zero_division=0)
            metrics_by_type[atk_type] = {
                "acc": acc,
                "precision": report.get("1", {}).get("precision", 0.0),
                "recall": report.get("1", {}).get("recall", 0.0),
                "f1-score": report.get("1", {}).get("f1-score", 0.0),
                "support": int(len(y_true_bin)),
            }
            if verbose:
                print(f"\n=== Attack Type {atk_type} (vs benign) ===")
                print(f"Accuracy: {acc:.4f}")
                print(classification_report(y_true_bin, y_pred_bin, zero_division=0))
                print("Confusion Matrix:")
                print(confusion_matrix(y_true_bin, y_pred_bin))
        return metrics_by_type
