import numpy as np
import torch


def distribute_batches_equally(data_loader, num_workers):
    """
    Gives each worker the same number of batches of data.

    :param data_loader: Data loader
    :type data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(data_loader):
        worker_idx = batch_idx % num_workers

        distributed_dataset[worker_idx].append((data, target))

    return distributed_dataset


def apply_data_replacement(X, Y, mal_data_loader, replacement_ratio):
    """
    Replace class labels using the replacement method

    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :param mal_data_loader: Mal data
    :type mal_data_loader: DataLoader
    :param replacement_ratio: Ratio to update sample
    :type replacement_method: float
    """
    X_mal = np.array(
        [tensor.numpy() for batch in mal_data_loader for tensor in batch[0]]
    )

    mal_id = np.random.randint(
        len(X_mal), size=round(len(X) * replacement_ratio))
    X_id = np.random.randint(len(X), size=round(len(X) * replacement_ratio))

    for id in range(len(mal_id)):
        X[X_id[id]] = X_mal[mal_id[id]]

    return (X, Y)


def apply_data_noise(X, Y, attack_noise_std, replacement_ratio):
    """
    Replace class labels using the replacement method

    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :param replacement_ratio: Ratio to update sample
    :type replacement_method: float
    """

    for id in range(round(len(X) * replacement_ratio)):
        X[id] = torch.add(
            torch.Tensor(X[id]),
            (attack_noise_std**0.5) * torch.rand_like(torch.Tensor(X[id])),
        )
    return (X, Y)


def poison_data(
    logger,
    distributed_dataset,
    num_workers,
    poisoned_worker_ids,
    noise_type,
    mal_data_loader,
    replacement_ratio,
    attack_noise_std,
):
    """
    Poison worker data

    :param logger: logger
    :type logger: loguru.logger
    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param num_workers: Number of workers overall
    :type num_workers: int
    :param poisoned_worker_ids: IDs poisoned workers
    :type poisoned_worker_ids: list(int)
    :param noise_type: type of noise (label_flipping or gaussian_noise)
    :type noise_type: string
    :param mal_data_loader: malicious data
    :type mal_data_loader: data loader
    :param replacement_method: Replacement methods to use to replace
    :type replacement_method: list(method)
    """
    poisoned_dataset = []
    logger.info("Poisoning data for workers: {}".format(
        str(poisoned_worker_ids)))
    # class_labels = list(set(distributed_dataset[0][1]))
    for worker_idx in range(num_workers):
        if worker_idx in poisoned_worker_ids:
            if noise_type == "label_flipping":
                poisoned_dataset.append(
                    apply_data_replacement(
                        distributed_dataset[worker_idx][0],
                        distributed_dataset[worker_idx][1],
                        mal_data_loader,
                        replacement_ratio,
                    )
                )
            else:
                poisoned_dataset.append(
                    apply_data_noise(
                        distributed_dataset[worker_idx][0],
                        distributed_dataset[worker_idx][1],
                        attack_noise_std,
                        replacement_ratio,
                    )
                )
        else:
            poisoned_dataset.append(distributed_dataset[worker_idx])
    return poisoned_dataset


def convert_distributed_data_into_numpy(distributed_dataset):
    """
    Converts a distributed dataset (returned by a data distribution method) from Tensors into numpy arrays.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    """
    converted_distributed_dataset = []

    def _to_numpy(item):
        """Safely convert tensors/arrays/scalars to numpy without assuming .numpy() exists."""
        # Torch tensor path
        if hasattr(item, "detach"):
            return item.detach().cpu().numpy()
        # Numpy array stays as is
        if isinstance(item, np.ndarray):
            return item
        # Fallback for numpy scalars / python scalars / lists
        return np.asarray(item)

    for worker_idx in range(len(distributed_dataset)):
        worker_data = distributed_dataset[worker_idx]

        X_ = np.array([
            _to_numpy(tensor)
            for batch in worker_data for tensor in batch[0]
        ])
        Y_ = np.array([
            _to_numpy(tensor)
            for batch in worker_data for tensor in batch[1]
        ])

        converted_distributed_dataset.append((X_, Y_))

    return converted_distributed_dataset
