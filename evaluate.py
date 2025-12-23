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
from function.utils import load_test_data_loader, load_train_data_loader
from function.utils.visualize_util import (
    collect_re_and_labels_from_client,
    collect_z_norms_and_labels_from_client,
    collect_latents_and_labels_from_client,
    collect_predictions_and_labels_from_client,
    plot_latent_embedding,
    plot_latent_embedding_non_iid_dir,
    plot_latent_first_component_hist_from_latents,
)
import json


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
        "by_attack_type": True if args_ns.experiment_type == "by_attack_type" else False,
        "experiment_type": args_ns.experiment_type,
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
        "-et",
        "--experiment_type",
        type=str,
        default="normal",
        choices=["normal", "by_attack_type", "non_iid_dir"],
        help="Experiment type: normal | by_attack_type | non_iid_dir",
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
        data_stage="test",
    )
    train_loaders = [None] * args.num_workers
    val_loaders = [None] * args.num_workers
    clients = create_clients(args, train_loaders, val_loaders, test_data_loaders)
    args.logger.info(
        "Testing {} model at epoch #{}", args.model_type, str(args_ns.epochs)
    )

    # Khi non_iid_dir: đọc meta phân chia từ train để biết seen_sets theo client
    if getattr(args_ns, 'experiment_type', 'normal') == 'non_iid_dir':
        try:
            setattr(args, 'partition_strategy', 'hybrid')
            if not hasattr(args, 'seen_per_client'):
                setattr(args, 'seen_per_client', 5)
            if not hasattr(args, 'dir_alpha'):
                setattr(args, 'dir_alpha', 1)
            if not hasattr(args, 'assign_seed'):
                setattr(args, 'assign_seed', 0)
            # num_workers cần khớp với số clients hiện tại
            setattr(args, 'num_workers', len(clients))
            train_data_loader_full = load_train_data_loader(logger, args)
            _ = client_data_process(
                args,
                train_data_loader_full,
                None,
                None,
                args.train_batch_size,
                poison=False,
                data_stage="train",
            )
            if hasattr(args, 'train_partition_meta'):
                args.logger.info(f"Loaded train partition meta: seen_sets={args.train_partition_meta.get('seen_sets', [])}")
            else:
                args.logger.warning("non_iid_dir: train_partition_meta not available; seen/unseen viz may be incomplete")
        except Exception as e:
            args.logger.warning(f"Failed to load train partition meta for non_iid_dir: {e}")

    multipliers = np.array([1.0])
    multiplier_auc_all = {m: [] for m in multipliers}

    summary_rows = []
    detailed_metrics_rows = []  # For seen/unseen/normal metrics
    
    # Global containers for analysis
    global_re = []
    global_labels = []
    global_client_idxs = []
    global_latent_firstcomp = []
    per_client_data = {}

    # Define output folder for global artifacts
    fallback_epoch_global = args_ns.epochs  # sẽ dùng epoch cuối cùng cho tên thư mục
    output_folder = os.path.join(
        "logs",
        "re_distributions",
        f"{args.model_type}_mc{args.num_multi_class_clients}_epoch_{fallback_epoch_global}",
    )
    os.makedirs(output_folder, exist_ok=True)

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
        
        # PTLAE dùng prototypes thay vì thresholds
        if args.model_type == "PTLAE":
            # Lấy proto_z0 và proto_z1 từ CSV
            proto_z0 = None
            proto_z1 = None
            
            if pd.notna(row.get("proto_z0")):
                try:
                    proto_z0 = eval(row["proto_z0"])
                except Exception:
                    proto_z0 = None
            
            if pd.notna(row.get("proto_z1")):
                try:
                    proto_z1 = eval(row["proto_z1"])
                except Exception:
                    proto_z1 = None
            
            args.logger.info(
                f"Client {client_idx} test at epoch {fallback_epoch} with proto_z0: {proto_z0 is not None}, proto_z1: {proto_z1 is not None}"
            )
            # Set prototypes cho client
            client.set_prototypes(proto_z0, proto_z1)
            # set_best_ckpt với threshold mặc định (không dùng cho PTLAE)
            client.set_best_ckpt(0, fallback_epoch, (0, 0), (0, 0), None)
        else:
            if args.model_type == "FedGH":
                # FedGH không lưu/không dùng threshold; chỉ log thông tin epoch
                args.logger.info(
                    f"Client {client_idx} test at epoch {fallback_epoch} (FedGH: thresholds not used)"
                )
            else:
                # Các phương pháp khác dùng thresholds
                threshold_re = (
                    eval(row["threshold_re"]) if pd.notna(row.get("threshold_re")) else (0, 0)
                )
                threshold_z = (
                    eval(row["threshold_z"]) if pd.notna(row.get("threshold_z")) else (0, 0)
                )
                args.logger.info(
                    f"Client {client_idx} test at epoch {fallback_epoch} with threshold re: {threshold_re}, and threshold z: {threshold_z}"
                )
                # FedHome no longer uses thresholds; load weights then record best_ckpt with current params

        # đọc và load lại mô hình cho từng client
        model_path = os.path.join(
            args_ns.model_dir, f"epoch_{fallback_epoch}_client_{client_idx}.pt"
        )
        client.update_nn_parameters(torch.load(model_path, map_location=client.device))
        if args.model_type == "FedHome":
            client.set_best_ckpt(0, fallback_epoch, client.get_nn_parameters())

        # ===== b) Collect Dữ Liệu Test =====
        # Collect per-sample RE and raw labels
        client_re_list, client_raw_labels = collect_re_and_labels_from_client(client)
        
        # Collect latent L2-norms
        client_z_list, client_z_labels = collect_z_norms_and_labels_from_client(client)
        
        # Collect full latent vectors
        latents, lat_labels = collect_latents_and_labels_from_client(client, max_samples_per_class=1000)

        # ===== c) Lưu Per-Client Artifacts =====
        out_dir = os.path.join(
            "logs",
            "re_distributions",
            f"{args.model_type}_mc{args.num_multi_class_clients}_epoch_{fallback_epoch}",
        )
        client_out_dir = os.path.join(out_dir, f"client_{client_idx}")
        os.makedirs(client_out_dir, exist_ok=True)

        # Per-client latent embedding (no prototype overlay)
        if latents is not None and latents.shape[0] > 0:
            # Plot latent embedding without prototypes
            # Use the checkpoint epoch we actually loaded
            plot_latent_embedding(latents, lat_labels, client_out_dir, fallback_epoch,
                                  client_id=client_idx, method='tsne')
            
            # Non_iid_dir specialized embedding
            if getattr(args, 'experiment_type', None) == 'non_iid_dir':
                # Lấy seen_set từ partition metadata
                pm_train = getattr(args, 'train_partition_meta', None) or {}
                seen_list = pm_train.get('seen_sets', []) if isinstance(pm_train, dict) else []
                seen_for_client = []
                if isinstance(seen_list, list) and client_idx < len(seen_list):
                    seen_for_client = list(map(int, seen_list[client_idx]))
                
                # Build attack labels
                num_attacks = getattr(args, 'num_attack_labels', None)
                if isinstance(num_attacks, int) and num_attacks > 0:
                    attack_labels = list(range(1, num_attacks + 1))
                else:
                    uniq = sorted(set(int(x) for x in lat_labels if int(x) != 0))
                    attack_labels = uniq
                
                # Vẽ 3-color plot (benign/attack_seen/attack_unseen)
                out = plot_latent_embedding_non_iid_dir(
                    latents, lat_labels, seen_for_client,
                    attack_label=attack_labels,
                    out_dir=client_out_dir,
                    epoch=fallback_epoch,
                    client_id=client_idx,
                    method='tsne',
                    max_points=1000,
                    random_state=getattr(args, 'assign_seed', 0)
                )
                if out:
                    args.logger.info(f"Saved non-iid-dir latent viz for client {client_idx} -> {out}")
                
                # Non_iid_dir histogram latent thành phần đầu
                hist_p = plot_latent_first_component_hist_from_latents(
                    latents, client_out_dir, fallback_epoch, client_id=client_idx
                )
                if hist_p:
                    args.logger.info(f"Saved client {client_idx} latent-dim0 histogram to {hist_p}")
                
                # Accumulate cho global histogram
                first_comp = np.asarray(latents)[:, 0].tolist()
                global_latent_firstcomp.extend(first_comp)

        # ===== d) Accumulate Global =====
        global_re.extend(client_re_list)
        global_labels.extend(client_raw_labels)
        global_client_idxs.extend([client_idx] * len(client_raw_labels))
        
        # Lưu per-client data để dùng cho seen/unseen analysis
        threshold_val = 0.0
        if args.model_type == "PTLAE":
            threshold_val = 0.0
        elif args.model_type == "FedGH":
            # FedGH không có threshold trong log; giữ placeholder 0.0
            threshold_val = 0.0
        else:
            threshold_re = eval(row["threshold_re"]) if pd.notna(row.get("threshold_re")) else (0, 0)
            threshold_val = float(threshold_re[0]) if isinstance(threshold_re, (list, tuple)) else float(threshold_re)
        
        per_client_data[client_idx] = {
            "re": list(map(float, client_re_list)),
            "labels": list(map(int, client_raw_labels)),
            "threshold_re": threshold_val,
        }

        # ===== e) test (như hiện tại) =====
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

        # ===== f) Collect predictions and compute detailed metrics =====
        if args_ns.experiment_type == "non_iid_dir":
            # Get predictions from client
            predictions, true_labels_raw = collect_predictions_and_labels_from_client(client)
            
            # Get seen_set for this client
            pm_train = getattr(args, 'train_partition_meta', None) or {}
            seen_list = pm_train.get('seen_sets', []) if isinstance(pm_train, dict) else []
            seen_for_client = set()
            if isinstance(seen_list, list) and client_idx < len(seen_list):
                seen_for_client = set(map(int, seen_list[client_idx]))
            
            # Categorize samples
            preds_arr = np.array(predictions, dtype=int)
            labels_arr = np.array(true_labels_raw, dtype=int)
            
            # Masks for different categories
            mask_normal = labels_arr == 0
            mask_attack_seen = np.array([lab in seen_for_client and lab != 0 for lab in labels_arr])
            mask_attack_unseen = np.array([lab not in seen_for_client and lab != 0 for lab in labels_arr])
            mask_attack_all = labels_arr != 0
            
            # Compute Recall_seen (TP rate for attack_seen)
            if mask_attack_seen.sum() > 0:
                recall_seen = (preds_arr[mask_attack_seen] == 1).sum() / mask_attack_seen.sum()
            else:
                recall_seen = np.nan
            
            # Compute Recall_unseen (TP rate for attack_unseen)
            if mask_attack_unseen.sum() > 0:
                recall_unseen = (preds_arr[mask_attack_unseen] == 1).sum() / mask_attack_unseen.sum()
            else:
                recall_unseen = np.nan
            
            # Compute Recall_normal (TN rate for normal)
            if mask_normal.sum() > 0:
                recall_normal = (preds_arr[mask_normal] == 0).sum() / mask_normal.sum()
            else:
                recall_normal = np.nan
            
            # Compute Precision_attack (correct attack predictions / all attack predictions)
            predicted_attack = preds_arr == 1
            if predicted_attack.sum() > 0:
                # Correct attack predictions = predicted 1 AND true label is attack
                correct_attack = predicted_attack & mask_attack_all
                precision_attack = correct_attack.sum() / predicted_attack.sum()
            else:
                precision_attack = np.nan
            
            # Compute Precision_normal (correct normal predictions / all normal predictions)
            predicted_normal = preds_arr == 0
            if predicted_normal.sum() > 0:
                # Correct normal predictions = predicted 0 AND true label is 0
                correct_normal = predicted_normal & mask_normal
                precision_normal = correct_normal.sum() / predicted_normal.sum()
            else:
                precision_normal = np.nan
            
            detailed_metrics_rows.append({
                "Client": f"Client {client_idx + 1}",
                "Recall_seen": recall_seen * 100 if not np.isnan(recall_seen) else np.nan,
                "Recall_unseen": recall_unseen * 100 if not np.isnan(recall_unseen) else np.nan,
                "Recall_normal": recall_normal * 100 if not np.isnan(recall_normal) else np.nan,
                "Precision_attack": precision_attack * 100 if not np.isnan(precision_attack) else np.nan,
                "Precision_normal": precision_normal * 100 if not np.isnan(precision_normal) else np.nan,
            })

    # ====== GLOBAL ANALYSIS (sau khi hoàn thành tất cả clients) ======
    # 1) Global histogram nếu non_iid_dir
    if args_ns.experiment_type == "non_iid_dir" and global_latent_firstcomp:
        # Reshape 1D array to 2D (N, 1) for the function
        arr = np.array(global_latent_firstcomp).reshape(-1, 1)
        global_hist_path = plot_latent_first_component_hist_from_latents(
            arr, 
            output_folder, 
            fallback_epoch_global, 
            client_id=None
        )
        if global_hist_path:
            args.logger.info(f"Saved global latent[0] histogram to {global_hist_path}")

    # 2) Per-attack AUC (nếu global_re có đủ data)
    if global_re and global_labels:
        from function.utils.visualize_util import compute_auc_per_attack_from_flat
        per_attack_auc_dict = compute_auc_per_attack_from_flat(
            global_labels,
            global_re
        )
        # Convert keys to strings for JSON compatibility
        per_attack_auc_json = {str(int(k)): (None if v is None else float(v)) for k, v in per_attack_auc_dict.items()}
        
        # Lưu per-attack AUC JSON
        per_attack_json_path = os.path.join(
            output_folder,
            f"{args.model_type}_per_attack_auc.json"
        )
        with open(per_attack_json_path, "w", encoding="utf-8") as f:
            json.dump(per_attack_auc_json, f, indent=4)
        args.logger.info(f"Saved per-attack AUC to {per_attack_json_path}")

        # Note: Per-client per-attack metrics and seen/unseen analysis require
        # additional implementation similar to serverPTL.py test_on_clients()

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
        # Tính global metrics
        global_auc = np.nanmean(df_summary["ROC-AUC"].values)
        global_precision = np.nanmean(df_summary["Precision"].values)
        global_recall = np.nanmean(df_summary["Recall"].values)
        global_f1 = np.nanmean(df_summary["F1"].values)
        # Compute ACC from precision and recall (or use harmonic mean as proxy)
        global_acc = (global_precision + global_recall) / 2.0  # Approximation
        
        # Log global metrics
        args.logger.info(
            "\n====== GLOBAL TEST METRICS (All Clients) ======\n"
            f"ROC-AUC:   {global_auc:.4f}\n"
            f"Precision: {global_precision:.4f}\n"
            f"Recall:    {global_recall:.4f}\n"
            f"F1:        {global_f1:.4f}\n"
            f"ACC:       {global_acc:.4f}\n"
        )
        
        avg_row = {
            "Client": "Average",
            "ROC-AUC": global_auc,
            "Precision": global_precision,
            "Recall": global_recall,
            "F1": global_f1,
        }
        df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)

        out_dir = os.path.dirname(log_csv_path)  # cùng thư mục logger/log csv
        os.makedirs(out_dir, exist_ok=True)
        base_name = f"{args.dataset}_{args.model_type}_epoch_{args_ns.epochs}_mul_{args.threshold_multiplier:.1f}"
        csv_path = os.path.join(out_dir, base_name + ".csv")
        df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    else:
        logger.warning("Không thu được thống kê nào tại multiplier yêu cầu.")

    # ===== Export detailed metrics CSV (seen/unseen/normal) =====
    if detailed_metrics_rows and args_ns.experiment_type == "non_iid_dir":
        df_detailed = pd.DataFrame(detailed_metrics_rows)
        
        # Compute global averages (nanmean to ignore NaN values)
        global_recall_seen = np.nanmean(df_detailed["Recall_seen"].values)
        global_recall_unseen = np.nanmean(df_detailed["Recall_unseen"].values)
        global_recall_normal = np.nanmean(df_detailed["Recall_normal"].values)
        global_precision_attack = np.nanmean(df_detailed["Precision_attack"].values)
        global_precision_normal = np.nanmean(df_detailed["Precision_normal"].values)
        
        # Log global detailed metrics
        args.logger.info(
            "\n====== GLOBAL DETAILED METRICS (Seen/Unseen/Normal) ======\n"
            f"Recall_seen:       {global_recall_seen:.4f}\n"
            f"Recall_unseen:     {global_recall_unseen:.4f}\n"
            f"Recall_normal:     {global_recall_normal:.4f}\n"
            f"Precision_attack:  {global_precision_attack:.4f}\n"
            f"Precision_normal:  {global_precision_normal:.4f}\n"
        )
        
        # Add global row to DataFrame
        global_row = {
            "Client": "All",
            "Recall_seen": global_recall_seen,
            "Recall_unseen": global_recall_unseen,
            "Recall_normal": global_recall_normal,
            "Precision_attack": global_precision_attack,
            "Precision_normal": global_precision_normal,
        }
        df_detailed = pd.concat([df_detailed, pd.DataFrame([global_row])], ignore_index=True)
        
        # Export to CSV
        out_dir = os.path.dirname(log_csv_path)
        os.makedirs(out_dir, exist_ok=True)
        base_name = f"{args.dataset}_{args.model_type}_epoch_{args_ns.epochs}_detailed_metrics"
        csv_path_detailed = os.path.join(out_dir, base_name + ".csv")
        df_detailed.to_csv(csv_path_detailed, index=False, encoding="utf-8-sig")
        args.logger.info(f"Saved detailed metrics CSV to {csv_path_detailed}")


if __name__ == "__main__":
    main()
