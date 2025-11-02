import numpy as np
from sklearn.cluster import KMeans

def average_nn_parameters(parameters):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum(
            [param[name].data for param in parameters]) / len(parameters)

    return new_params


def attention_average_nn_parameters(parameters, attentions):
    """
    Averages based on attention passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    :param attentions: attention-based aggregation coefficient
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum(
            [param[name].data * coef for (param, coef) in zip(parameters, attentions)])

    return new_params

def kmeans_cluster_parameters(parameters, n_clusters=2):
    """
    Cluster client parameters into `n_clusters` using K-means and compute the average for each cluster.

    :param parameters: List of dictionaries representing parameters of the neural network.
    :type parameters: list
    :param n_clusters: Number of clusters for K-means.
    :type n_clusters: int
    :return: A dictionary containing:
             - `clustered_params`: A list of averaged parameters for each cluster.
             - `cluster_assignments`: A list of cluster labels corresponding to each parameter set.
    :rtype: dict
    """
    # Convert parameters to a flattened list of vectors for clustering
    flattened_params = [
        np.concatenate([param[name].data.cpu().numpy().flatten() for name in param.keys()])
        for param in parameters
    ]


    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(flattened_params)

    # Organize parameters by cluster
    clustered_params = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(cluster_labels):
        clustered_params[label].append(parameters[idx])

    # Compute average parameters for each cluster
    averaged_cluster_params = [
        average_nn_parameters(cluster) for cluster in clustered_params
    ]

    return {
        "clustered_params": averaged_cluster_params,
        "cluster_assignments": cluster_labels
    }


def kmeans_cluster_parameters_and_get_min_center(parameters, list_loss, K):
    """
    Cluster parameters based on the given loss functions using K-means algorithm.
    Print the center of the cluster with smallest center.

    :param parameters: List of dictionaries representing parameters of the neural network.
    :type parameters: list
    :param list_loss: List of loss functions corresponding to the list of dictionaries above.
    :type list_loss: list
    :param K: Number of clusters.
    :type K: int
    :return: List of dictionaries representing parameters of the neural network in the cluster with smallest center.
    :rtype: list
    """
    if K <= 0 or K >= len(list_loss):
        return parameters

    # Convert list_loss to numpy array for easier manipulation
    np_list_loss = np.asarray(list_loss, dtype=np.float32)

    # Reshape list_loss to column vector
    np_list_loss_reshaped = np_list_loss.reshape(-1, 1)

    # Use KMeans clustering
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(np_list_loss_reshaped)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Print cluster centers
    #print("Cluster Centers:")
    #for i, center in enumerate(cluster_centers):
        #print(f"Cluster {i + 1} center:", center)

    # Find cluster with smallest center
    min_center_idx = np.argmin(np.linalg.norm(cluster_centers, axis=1))

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Find indices of parameters in the cluster with smallest center
    min_center_indices = np.where(cluster_labels == min_center_idx)[0]

    # Get parameters in the cluster with smallest center
    min_center_params = [parameters[idx] for idx in min_center_indices]

    return min_center_params


def get_worker_num_from_model_file_name(model_file_name):
    """
    :param model_file_name: string
    """
    return int(model_file_name.split("_")[1])


def get_epoch_num_from_model_file_name(model_file_name):
    """
    :param model_file_name: string
    """
    return int(model_file_name.split("_")[2].split(".")[0])


def get_suffix_from_model_file_name(model_file_name):
    """
    :param model_file_name: string
    """
    return model_file_name.split("_")[3].split(".")[0]


def get_model_files_for_worker(model_files, worker_id):
    """
    :param model_files: list[string]
    :param worker_id: int
    """
    worker_model_files = []

    for model in model_files:
        worker_num = get_worker_num_from_model_file_name(model)

        if worker_num == worker_id:
            worker_model_files.append(model)

    return worker_model_files


def get_model_files_for_epoch(model_files, epoch_num):
    """
    :param model_files: list[string]
    :param epoch_num: int
    """
    epoch_model_files = []

    for model in model_files:
        model_epoch_num = get_epoch_num_from_model_file_name(model)

        if model_epoch_num == epoch_num:
            epoch_model_files.append(model)

    return epoch_model_files


def get_model_files_for_suffix(model_files, suffix):
    """
    :param model_files: list[string]
    :param suffix: string
    """
    suffix_only_model_files = []

    for model in model_files:
        model_suffix = get_suffix_from_model_file_name(model)

        if model_suffix == suffix:
            suffix_only_model_files.append(model)

    return suffix_only_model_files
