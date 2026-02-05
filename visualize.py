import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
plt.rcParams["font.family"] = "serif"

import argparse

def visualize(file_name, prefix, interval):
    # # Ensure directory for visualizations exists
    # visualize_dir = "visualizes"
    # os.makedirs(visualize_dir, exist_ok=True)

    # # Read the CSV file
    # # file_path = f"logs/{file_name}"
    # df = pd.read_csv(file_name)

    # # Prepare xlsx files for each loss type
    # metrics = ['train_re', 'train_ptl_loss']
    # if 'val_roc_auc' in df.columns:  # Check if 'val_roc_auc' exists
    #     metrics.append('val_roc_auc')

    # for metric in metrics:
    #     # Filter epochs divisible by the interval
    #     filtered_df = df[df['epoch'] % interval == 0]

    #     # Create a pivot table for each loss type
    #     pivot_data = filtered_df.pivot_table(
    #         index='epoch',
    #         columns='client_id',
    #         values=metric,
    #         aggfunc='first'
    #     )
    #     output_xlsx = os.path.join(visualize_dir, f"{prefix}_{interval}_{metric}.xlsx")
    #     pivot_data.to_excel(f"{prefix}_{interval}_{metric}.xlsx")

    # # Calculate and visualize average losses for each epoch
    # filtered_epochs = sorted(df[df['epoch'] % interval == 0]['epoch'].unique())
    # results = {metric: {'malicious': [], 'non_malicious': [], 'all': []} for metric in metrics}


    # for epoch in filtered_epochs:
    #     epoch_data = df[df['epoch'] == epoch]
    #     for metric in metrics:
    #         mal_clients = epoch_data[epoch_data['is_mal'] == True]
    #         non_mal_clients = epoch_data[epoch_data['is_mal'] == False]

    #         results[metric]['malicious'].append(mal_clients[metric].mean())
    #         results[metric]['non_malicious'].append(non_mal_clients[metric].mean())
    #         results[metric]['all'].append(epoch_data[metric].mean())

    # # Plot the results
    # for metric in metrics:
    #     plt.figure(figsize=(20, 12))
    #     plt.plot(filtered_epochs, results[metric]['malicious'], label="Malicious Clients", marker='o')
    #     plt.plot(filtered_epochs, results[metric]['non_malicious'], label="Non-Malicious Clients", marker='o')
    #     plt.plot(filtered_epochs, results[metric]['all'], label="All Clients", marker='o')

    #     plt.title(f"{prefix} Average {metric} per {interval} Epoch")
    #     plt.xlabel("Epoch")
    #     plt.ylabel(f"{metric}")
    #     plt.legend()
    #     plt.grid()
    #     output_png = os.path.join(visualize_dir, f"{prefix}_{interval}_{metric}_plot.png")
    #     plt.savefig(f"{prefix}_{interval}_{metric}_plot.png")
    #     plt.close()


    # Ensure directory for visualizations exists
    visualize_dir = "visualizes"
    os.makedirs(visualize_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(file_name)

    # Metrics to visualize depend on available columns
    if 'train_ptl_loss' in df.columns:
        metrics = ['train_re', 'train_ptl_loss']  # PTLAE (or any model logging PTL loss)
    elif 'train_latent_z' in df.columns:
        metrics = ['train_re', 'train_latent_z']  # other models
    else:
        metrics = ['train_re']

    for metric in metrics:
        plt.figure(figsize=(40, 24))
        
        # Loop through all clients and plot their metrics over epochs
        for client_id in df['client_id'].unique():
            client_data = df[df['client_id'] == client_id]
            label = f"Client {client_id}"
            
            # Determine line color based on whether the client is malicious
            color = 'red' if client_data['is_mal'].iloc[0] else 'blue'
            
            plt.plot(
                client_data['epoch'], 
                client_data[metric], 
                label=label if color == 'red' else None,  # Only label malicious clients
                color=color, 
                linewidth=1, 
                alpha=0.8,
                marker='o'
            )

        # Add plot details
        plt.title(f"{prefix} {metric} for All Clients")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(loc='upper right', fontsize='small', title="Malicious Clients")
        plt.tight_layout()
        
        # Save the plot (create nested dirs if prefix contains subfolders)
        output_png = os.path.join(visualize_dir, f"{prefix}_{interval}_{metric}_plot.png")
        output_dir = os.path.dirname(output_png)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_png)
        plt.close()

def visualize_loss_by_client(path_folder, filename_prefix="", interval=1, max_epoch=4000):
    """
    Visualize loss (train_loss) for each client over epochs, from multiple *_train.csv files in a folder.

    Args:
        path_folder (str): Folder path containing *_train.csv files.
        filename_prefix (str): Optional prefix to filter files.
        interval (int): Interval of epochs to include (default = 1).
        max_epoch (int): Max epoch to plot (default = 4000).
    """
    os.makedirs("visualizes", exist_ok=True)

    # Get all matching CSV files
    pattern = os.path.join(path_folder, f"{filename_prefix}*_train.csv")
    csv_files = sorted(glob(pattern))

    valid_datasets = ["cic_ids", "ctu13_08", "unsw", "ton_iot_network", "nb_iot", "wsn_ds"]

    for csv_file in csv_files:
        try:
            # Get dataset name from filename
            base_name = os.path.basename(csv_file)
            dataset_name = next((ds for ds in valid_datasets if base_name.startswith(ds + "_")), None)
            print(f"Processing file: {csv_file} for dataset: {dataset_name}")
            if dataset_name not in valid_datasets:
                print(f"Skipping unknown dataset: {dataset_name}")
                continue

            df = pd.read_csv(csv_file)

            # Filter by epoch
            df = df[df['epoch'] % interval == 0]
            df = df[df['epoch'] <= max_epoch]

            dataset_dir = os.path.join("visualizes/DualLossAE2", dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            # Loop over all clients
            for client_id in df['client_id'].unique():
                client_data = df[df['client_id'] == client_id]
                # client_data = client_data[~((client_data['epoch'] > 200) & (client_data['train_loss'] > 0.584))]


                plt.figure(figsize=(4, 2.5))
                plt.plot(
                    client_data['epoch'],
                    client_data['train_loss'],
                    linewidth=1.2,
                    label=f"Client {client_id}",
                    color='red' if client_data['is_mal'].iloc[0] else 'blue'
                )

                plt.xlabel("Epoch",  fontsize=7)
                plt.ylabel("Training Loss",  fontsize=7)
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)

                plt.title(f"{dataset_name.upper()} - Client {client_id}",  fontsize=8)
                plt.grid(True)
                plt.tight_layout()

                save_path = os.path.join(dataset_dir, f"client{int(client_id)}.png")
                plt.savefig(save_path, dpi=200)
                plt.close()

        except Exception as e:
            print(f"[ERROR] Could not process file {csv_file}: {e}")



def plot_mse_seen_unseen(csv_path, title_prefix="DualLossAE"):
    """
    Plot mean MSE over epochs for normal traffic vs seen / unseen attacks.

    Args:
        csv_path (str): Path to mse_seen_unseen.csv produced during testing.
        title_prefix (str): Prefix for saved figure name.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"epoch", "group", "mean_mse"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required_cols}")

    os.makedirs("visualizes", exist_ok=True)

    plt.figure(figsize=(6, 4))
    for group in ["normal", "seen", "unseen"]:
        if group not in df["group"].unique():
            continue
        sub = df[df["group"] == group].sort_values("epoch")
        plt.plot(sub["epoch"], sub["mean_mse"], marker='o', linewidth=1.5, label=group)

    plt.title(f"{title_prefix}: MSE by group")
    plt.xlabel("Epoch")
    plt.ylabel("Mean MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join("visualizes", f"{title_prefix}_mse_seen_unseen.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved MSE plot -> {out_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Visualize Loss Data")
    # parser.add_argument("-file", type=str, required=True, help="CSV file to visualize")
    # parser.add_argument("-prefix", type=str, required=True, help="Prefix for output files and plots")
    # parser.add_argument("-interval", type=int, required=True, help="Interval of epochs to filter")
    # args = parser.parse_args()

    # visualize(args.file, args.prefix, args.interval)

    visualize_loss_by_client(path_folder="logs/DualLossAE2", filename_prefix="unsw", interval=1,max_epoch=450)

