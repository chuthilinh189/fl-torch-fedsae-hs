from datetime import datetime
import numpy as np
import random

def generate_experiment_ids(
    dataset,model_type, epochs, replacement_ratio, num_poisoned_workers, attack_noise_std, aggregation_type, threshold_multiplier, num_multi_class_clients
):
    """
    Generate the filename for experiment ID.

    :param model_type: index for experiments
    :param replacement_ratio: ratio
    :param num_poisoned_workers: number of poison client
    """
    if num_poisoned_workers == 0:
        log_file = "logs/{}_{}_{}_{}_{}_multi{}_{}.log".format(
            dataset,
            model_type,
            str(epochs),
            str(num_poisoned_workers),
            str(aggregation_type),
            str(threshold_multiplier),
            str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S")),
            str(num_multi_class_clients),
        )
    else:
        log_file = "logs/{}_{}_{}_{}_{}_multi{}_{}_{}_{}.log".format(
            dataset,
            model_type,
            str(epochs),
            str(num_poisoned_workers),
            str(aggregation_type),
            str(threshold_multiplier),
            str(replacement_ratio),
            str(attack_noise_std),
            str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S")),
            str(num_multi_class_clients),
        )

    return log_file

# def create_dev_dataset_from_clients(clients, max_samples_per_client=50):
#     dev_samples = []

#     for client in clients:
#         # Truy cập vào data của client (tập train)
#         count = 0
#         for x, y in client.train_data_loader:
#             # Chỉ lấy mẫu có nhãn là 0 (bình thường)
#             for xi, yi in zip(x, y):
#                 if yi.item() == 0:
#                     dev_samples.append(xi.numpy())
#                     count += 1
#                     if count >= max_samples_per_client:
#                         break
#             if count >= max_samples_per_client:
#                 break

#     # Kết quả: numpy array [num_clients * max_samples, feature_dim]
#     dev_dataset = np.array(dev_samples, dtype=np.float32)
#     return dev_dataset

def create_dev_dataset_from_clients(clients, max_samples_per_client=100):
    dev_samples = []

    for client in clients:
        all_data = []

        # Thu thập toàn bộ tensor từ train_data_loader
        for x, _ in client.train_data_loader:
            for xi in x:
                all_data.append(xi.numpy())

        # Lấy ngẫu nhiên max_samples_per_client mẫu
        if len(all_data) > max_samples_per_client:
            selected = random.sample(all_data, max_samples_per_client)
        else:
            selected = all_data

        dev_samples.extend(selected)

    dev_dataset = np.array(dev_samples, dtype=np.float32)
    return dev_dataset
