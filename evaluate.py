import os
import torch
import argparse
import pandas as pd
from loguru import logger
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import glob

from function.arguments import Arguments
from agents.clients import get_client_class
from core.data_processing import client_data_process
from core.client_factory import create_clients
from function.utils import load_test_data_loader


def build_config(args_ns):
    return {
        "dataset": args_ns.dataset,
        "train_batch_size": args_ns.train_batch_size,
        "val_batch_size": args_ns.val_batch_size,
        "test_batch_size": 1,
        "mal_batch_size": args_ns.val_batch_size,
        "dimension": args_ns.dimension,
        "epochs": args_ns.epochs,
        "model_type": args_ns.model_type,
        "noise_type": args_ns.noise_type,
        "num_of_poisoned_workers": args_ns.poisoned_workers,
        "poisoned_sample_ratio": args_ns.poisoned_ratio,
        "learning_rate": args_ns.learning_rate,
        "noise_std": args_ns.noise_std,
        "attack_noise_std": args_ns.attack_noise_std,
        "aggregation_type": args_ns.aggregation_type,
        "coef_shrink_ae": args_ns.coef_shrink_ae,
        "threshold_multiplier": args_ns.threshold_multiplier,
        "num_multi_class_clients": args_ns.num_multi_class_clients,
        "by_attack_type": args_ns.by_attack_type,
    }


def find_latest_valid_epoch(log_df, client_id, target_epoch):
    valid_rows = log_df[
        (log_df["client_id"] == client_id) & (log_df["epoch"] <= target_epoch)
    ]
    valid_rows = valid_rows[(valid_rows["epoch"] - 1) % 10 == 0]
    if valid_rows.empty:
        return None
    return valid_rows["epoch"].max() - 1


def find_log_file(log_prefix: str) -> str:
    matched_files = glob.glob(f"{log_prefix}*train.csv")
    if not matched_files:
        raise FileNotFoundError(
            f"No log CSV file found with prefix: {log_prefix} and suffix '*train.csv'"
        )
    if len(matched_files) > 1:
        raise ValueError(
            f"Multiple log files found with prefix '{log_prefix}': {matched_files}"
        )
    return matched_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved models with thresholds from log"
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="Dataset name (e.g., cic_ids, nslkdd)"
    )
    parser.add_argument(
        "-m",
        "--model_type",
        required=True,
        help="Model type (AE, DAE, SAE, SDAE, SupAE, DualLossAE)",
    )
    parser.add_argument(
        "-tbs", "--train_batch_size", type=int, default=128, help="Train batch size"
    )
    parser.add_argument(
        "-vbs", "--val_batch_size", type=int, default=128, help="Val batch size"
    )
    parser.add_argument("-di", "--dimension", type=int, default=128, help="Dimension")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("-ep", "--epochs", type=int, default=4000, help="Epochs")
    parser.add_argument(
        "-agg",
        "--aggregation_type",
        type=str,
        default="average",
        help="Aggregation type",
    )
    parser.add_argument(
        "-nt", "--noise_type", type=str, default="label_flipping", help="Noise type"
    )
    parser.add_argument(
        "-pw",
        "--poisoned_workers",
        type=int,
        default=0,
        help="Number of poisoned clients",
    )
    parser.add_argument(
        "-pr", "--poisoned_ratio", type=float, default=1.0, help="Poisoned sample ratio"
    )
    parser.add_argument(
        "-ns", "--noise_std", type=float, default=0.001, help="Noise stddev"
    )
    parser.add_argument(
        "-ans",
        "--attack_noise_std",
        type=float,
        default=3.0,
        help="Attack noise stddev",
    )
    parser.add_argument(
        "-cs", "--coef_shrink_ae", type=float, default=1.0, help="Coefficient shrink AE"
    )
    parser.add_argument(
        "-tm",
        "--threshold_multiplier",
        type=float,
        default=0.0,
        help="Threshold multiplier",
    )
    parser.add_argument(
        "-mc",
        "--num_multi_class_clients",
        type=int,
        default=0,
        help="Multi-class client count",
    )
    parser.add_argument(
        "-at",
        "--by_attack_type",
        type=bool,
        default=False,
        help="By attack type (False/ True)",
    )
    parser.add_argument(
        "--model_dir", required=True, help="Directory containing saved models"
    )
    parser.add_argument(
        "--log_csv",
        required=True,
        help="Prefix of CSV log file (e.g., logs/unsw_DualLossAE_)",
    )

    args_ns = parser.parse_args()
    config = build_config(args_ns)
    args = Arguments(logger, config)
    log_csv_path = find_log_file(args_ns.log_csv)
    df_log = pd.read_csv(log_csv_path)

    # tạo client và chuẩn bị dữ liệu test
    test_data_loader = load_test_data_loader(logger, args)
    test_data_loaders = client_data_process(
        args,
        test_data_loader,
        None,
        None,
        args.test_batch_size,
        poison=False,
    )
    train_loaders = [None] * args.num_workers
    val_loaders = [None] * args.num_workers
    clients = create_clients(args, train_loaders, val_loaders, test_data_loaders)
    args.logger.info(
        "Testing {} model at epoch #{}", args.model_type, str(args_ns.epochs)
    )

    multipliers = np.arange(0.0, 5.1, 0.2)
    multiplier_auc_all = {m: [] for m in multipliers}

    summary_rows = []

    for client_idx, client in enumerate(clients):

        # đọc và cập nhật ngưỡng của từng client
        fallback_epoch = args_ns.epochs
        row = df_log[
            (df_log["epoch"] == args_ns.epochs) & (df_log["client_id"] == client_idx)
        ]
        if row.empty:
            fallback_epoch = find_latest_valid_epoch(
                df_log, client_idx, args_ns.epochs - 1
            )
            if fallback_epoch is None:
                logger.warning(
                    f"⚠️ No valid log info for client {client_idx + 1} up to epoch {args_ns.epochs}"
                )
                continue
            logger.warning(
                f"↩️ Falling back to epoch {fallback_epoch} for client {client_idx + 1}"
            )
            row = df_log[
                (df_log["epoch"] == fallback_epoch)
                & (df_log["client_id"] == client_idx)
            ]
        row = row.iloc[0]
        threshold_re = (
            eval(row["threshold_re"]) if pd.notna(row["threshold_re"]) else (0, 0)
        )
        threshold_z = (
            eval(row["threshold_z"]) if pd.notna(row["threshold_z"]) else (0, 0)
        )
        args.logger.info(f"Client {client_idx} test at epoch {fallback_epoch} with threshold re: {threshold_re}, and thredhold z: {threshold_z}")
        client.set_best_ckpt(0, fallback_epoch, threshold_re, threshold_z, None)

        # đọc và load lại mô hình cho từng client
        model_path = os.path.join(
            args_ns.model_dir, f"epoch_{fallback_epoch}_client_{client_idx}.pt"
        )
        client.update_nn_parameters(torch.load(model_path, map_location=client.device))

        # test
        acc_list, precision_list, recall_list, f1_list, auc_list = client.test()
        for i, m in enumerate(multipliers):
            acc, precision, recall, f1, auc = (
                acc_list[i],
                precision_list[i],
                recall_list[i],
                f1_list[i],
                auc_list[i],
            )

            if abs(m - args.threshold_multiplier) < 1e-4:
                args.logger.info(
                    f"[Client {client_idx + 1}] Multiplier {m:.1f}: ACC={acc:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
                )

                # Lưu đúng chỉ số tại multiplier yêu cầu
                summary_rows.append(
                    {
                        "Client": f"Client {client_idx + 1}",
                        "ROC-AUC": float(auc * 100),
                        "Precision": float(precision * 100),
                        "Recall": float(recall * 100),
                        "F1": float(f1 * 100),
                    }
                )

            multiplier_auc_all[m].append(auc)

    # Tổng họpw kết quả test
    header = "\n====== AVERAGE AUC PER MULTIPLIER ======\n"
    header += "{:<12} {:<20} \n".format("Multiplier", "All Clients")
    header += "-" * 75 + "\n"
    rows = ""
    for m in sorted(multiplier_auc_all.keys()):
        all_avg = np.mean(multiplier_auc_all[m]) if multiplier_auc_all[m] else 0

        rows += "{:<12.1f} {:<20.6f} \n".format(m, all_avg)

    args.logger.info(header + rows)

    # ===== Xuất bảng thống kê tại multiplier = args.threshold_multiplier =====
    if summary_rows:
        df_summary = pd.DataFrame(
            summary_rows, columns=["Client", "ROC-AUC", "Precision", "Recall", "F1"]
        )
        avg_row = {
            "Client": "Average",
            "ROC-AUC": np.nanmean(df_summary["ROC-AUC"].values),
            "Precision": np.nanmean(df_summary["Precision"].values),
            "Recall": np.nanmean(df_summary["Recall"].values),
            "F1": np.nanmean(df_summary["F1"].values),
        }
        df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)

        out_dir = os.path.dirname(log_csv_path)  # cùng thư mục logger/log csv
        os.makedirs(out_dir, exist_ok=True)
        base_name = f"{args.dataset}_{args.model_type}_epoch_{args_ns.epochs}_mul_{args.threshold_multiplier:.1f}"
        csv_path = os.path.join(out_dir, base_name + ".csv")
        df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    else:
        logger.warning("Không thu được thống kê nào tại multiplier yêu cầu.")


if __name__ == "__main__":
    main()
