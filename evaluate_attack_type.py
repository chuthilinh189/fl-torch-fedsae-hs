import os
import torch
import argparse
import pandas as pd
from loguru import logger
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import glob
from collections import defaultdict

from function.arguments import Arguments
from agents.clients import get_client_class
from core.data_processing import client_data_process
from core.client_factory import create_clients
from function.utils import load_test_data_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved models with thresholds from log")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name (e.g., cic_ids, nslkdd)")
    parser.add_argument("-m", "--model_type", required=True, help="Model type (AE, DAE, SAE, SDAE, SupAE, DualLossAE, PTL)")
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=128, help="Train batch size")
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=128, help="Val batch size")
    parser.add_argument("-di", "--dimension", type=int, default=128, help="Dimension")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-ep", "--epochs", type=int, default=4000, help="Epochs")
    parser.add_argument("-agg", "--aggregation_type", type=str, default="average", help="Aggregation type")
    parser.add_argument("-nt", "--noise_type", type=str, default="label_flipping", help="Noise type")
    parser.add_argument("-pw", "--poisoned_workers", type=int, default=0, help="Number of poisoned clients")
    parser.add_argument("-pr", "--poisoned_ratio", type=float, default=1.0, help="Poisoned sample ratio")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.001, help="Noise stddev")
    parser.add_argument("-ans", "--attack_noise_std", type=float, default=3.0, help="Attack noise stddev")
    parser.add_argument("-cs", "--coef_shrink_ae", type=float, default=1.0, help="Coefficient shrink AE")
    parser.add_argument("-tm", "--threshold_multiplier", type=float, default=0.0, help="Threshold multiplier")
    parser.add_argument("-mc", "--num_multi_class_clients", type=int, default=0, help="Multi-class client count")
    parser.add_argument("-at", "--by_attack_type", type=bool, default=False, help="By attack type (False/ True)")
    parser.add_argument("--model_dir", required=True, help="Directory containing saved models")
    parser.add_argument("--log_csv", required=True, help="Prefix of CSV log file (e.g., logs/unsw_DualLossAE_)")
    return parser.parse_args()

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

def find_log_file(log_prefix: str) -> str:
    matched_files = glob.glob(f"{log_prefix}*train.csv")
    if not matched_files:
        raise FileNotFoundError(f"No log CSV file found with prefix: {log_prefix} and suffix '*train.csv'")
    if len(matched_files) > 1:
        raise ValueError(f"Multiple log files found with prefix '{log_prefix}': {matched_files}")
    return matched_files[0]

def find_latest_valid_epoch(log_df, client_id, target_epoch):
    valid_rows = log_df[(log_df["client_id"] == client_id) & (log_df["epoch"] <= target_epoch )] 
    valid_rows = valid_rows[(valid_rows["epoch"] - 1) % 10 == 0]
    if valid_rows.empty:
        return None
    return valid_rows["epoch"].max() - 1

def main():
    args_ns = parse_args()
    config = build_config(args_ns)
    args = Arguments(logger, config)
    args.by_attack_type = True

    log_csv_path = find_log_file(args_ns.log_csv)
    df_log = pd.read_csv(log_csv_path)

    test_data_loader = load_test_data_loader(logger, args)
    test_data_loaders = client_data_process(
        args, test_data_loader, None, None, args.test_batch_size, poison=False
    )
    # ensure stage awareness for full coverage in test
    test_data_loaders = client_data_process(
        args, test_data_loader, None, None, args.test_batch_size, poison=False, data_stage="test"
    )
    clients = create_clients(args, [None]*args.num_workers, [None]*args.num_workers, test_data_loaders)

    logger.info("🔍 Evaluate ACC per attack type for model: {}, epoch: {}", args.model_type, args_ns.epochs)

    summary_by_type = defaultdict(list)  # {attack_type: [(acc, support), ...]}

    for client_idx, client in enumerate(clients):
        # đọc log threshold
        fallback_epoch = args_ns.epochs
        row = df_log[(df_log["epoch"] == args_ns.epochs) & (df_log["client_id"] == client_idx)]
        if row.empty:
            fallback_epoch = find_latest_valid_epoch(df_log, client_idx, args_ns.epochs - 1)
            if fallback_epoch is None:
                logger.warning(f"⚠️ No valid log info for client {client_idx + 1} up to epoch {args_ns.epochs}")
                continue
            logger.warning(f"↩️ Falling back to epoch {fallback_epoch} for client {client_idx + 1}")
            row = df_log[(df_log["epoch"] == fallback_epoch) & (df_log["client_id"] == client_idx)]
        row = row.iloc[0]
        threshold_re = eval(row["threshold_re"]) if pd.notna(row["threshold_re"]) else (0, 0)
        threshold_z = eval(row["threshold_z"]) if pd.notna(row["threshold_z"]) else (0, 0)
        client.set_best_ckpt(0, fallback_epoch, threshold_re, threshold_z, None)

        # load lại mô hình
        model_path = os.path.join(args_ns.model_dir, f"epoch_{fallback_epoch}_client_{client_idx}.pt")
        client.update_nn_parameters(torch.load(model_path, map_location=client.device))

        # test theo attack type
        results = client.test_by_attack_type_full(threshold_re[0] + args.threshold_multiplier * threshold_re[1], threshold_z[0] + args.threshold_multiplier * threshold_z[1], verbose=False)

        logger.info(f"\n📊 Client {client_idx + 1} - Accuracy by Attack Type")
        logger.info(f"{'Attack Type':<15} {'ACC':>8} {'Support':>8}")
        for atk_type, metric in sorted(results.items()):
            acc = metric["acc"]
            support = metric["support"]
            logger.info(f"{atk_type:<15} {acc:>8.4f} {support:>8}")
            summary_by_type[atk_type].append((acc, support))


    # === Tổng kết toàn bộ clients ===
    logger.info("\n===== AVERAGE ACCURACY BY ATTACK TYPE (WEIGHTED) =====")
    logger.info(f"{'Attack Type':<15} {'Weighted ACC':>15} {'Total Samples':>15} {'#Clients':>10}")

    overall_weighted_sum = 0.0
    overall_total_support = 0

    for atk_type in sorted(summary_by_type.keys()):
        accs = [acc for acc, _ in summary_by_type[atk_type]]
        supports = [support for _, support in summary_by_type[atk_type]]

        total_support = sum(supports)
        num_clients = len(accs)

        if total_support > 0:
            weighted_avg = np.average(accs, weights=supports)
        else:
            weighted_avg = 0.0

        overall_weighted_sum += weighted_avg * total_support
        overall_total_support += total_support

        logger.info(f"{atk_type:<15} {weighted_avg:>15.4f} {sum(supports):>15} {len(accs):>10}")

    logger.info("\n📌 OVERALL WEIGHTED ACCURACY ACROSS ALL CLIENTS & ATTACK TYPES:")
    if overall_total_support > 0:
        overall_weighted_acc = overall_weighted_sum / overall_total_support
        print(f"{'Overall ACC':<20}: {overall_weighted_acc:.4f}")
    else:
        print("⚠️ No support data available. Cannot compute overall accuracy.")


    

if __name__ == "__main__":
    main()
