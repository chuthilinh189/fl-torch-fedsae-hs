import numpy
import os
import pickle
import random
from ..datasets import Dataset
import torch
from torch.utils.data import DataLoader, TensorDataset


def filter_data_loader_by_label(data_loader, label_to_keep=0):
    """
    Lọc DataLoader để chỉ giữ lại các mẫu có nhãn bằng label_to_keep.
    """
    X_list = []
    Y_list = []
    # Duyệt qua tất cả các batch của data_loader
    for batch in data_loader:
        data, target = batch
        # Tạo mask chỉ giữ mẫu có nhãn bằng label_to_keep
        mask = (target == label_to_keep)
        if mask.sum().item() > 0:
            X_list.append(data[mask])
            Y_list.append(target[mask])
    if len(X_list) == 0:
        raise ValueError("Không có mẫu nào thỏa mãn điều kiện lọc")
    # Nối các tensor lại với nhau
    X_all = torch.cat(X_list, dim=0)
    Y_all = torch.cat(Y_list, dim=0)
    # Tạo dataset và DataLoader mới
    dataset = TensorDataset(X_all, Y_all)
    new_loader = DataLoader(dataset, batch_size=data_loader.batch_size, shuffle=False)
    return new_loader


def generate_data_loaders_from_distributed_dataset(distributed_dataset, batch_size):
    """
    Generate data loaders from a distributed dataset.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param batch_size: batch size for data loader
    :type batch_size: int
    """
    data_loaders = []
    for worker_data in distributed_dataset:
        data_loaders.append(
            Dataset.get_data_loader_from_data(
                batch_size,
                worker_data[0],
                worker_data[1],
                shuffle=False,
            )
        )

    return data_loaders


def load_train_data_loader(logger, args):
    """
    Loads the training data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if args.by_attack_type:
        if os.path.exists(args.train_data_loader_by_attack_type_pickle_path):
            return load_data_loader_from_file(logger, args.train_data_loader_by_attack_type_pickle_path)
        else:
            logger.error("Couldn't find train data loader by attack type stored in file")

            raise FileNotFoundError(
                "Couldn't find train data loader by attack type stored in file")
    else:
        if os.path.exists(args.train_data_loader_pickle_path):
            return load_data_loader_from_file(logger, args.train_data_loader_pickle_path)
        else:
            logger.error("Couldn't find train data loader stored in file")

            raise FileNotFoundError(
                "Couldn't find train data loader stored in file")


def generate_train_loader(args, dataset):
    train_dataset = dataset.get_train_dataset()
    X, Y = shuffle_data(train_dataset)

    return dataset.get_data_loader_from_data(args.train_batch_size, X, Y)


def load_val_data_loader(logger, args):
    """
    Loads the validation data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if args.by_attack_type:
        if os.path.exists(args.val_data_loader_by_attack_type_pickle_path):
            return load_data_loader_from_file(logger, args.val_data_loader_by_attack_type_pickle_path)
        else:
            logger.error("Couldn't find val data loader by attack type stored in file")

            raise FileNotFoundError(
                "Couldn't find val data loader by attack type stored in file")
    else:
        if os.path.exists(args.val_data_loader_pickle_path):
            return load_data_loader_from_file(logger, args.val_data_loader_pickle_path)
        else:
            logger.error("Couldn't find val data loader stored in file")

            raise FileNotFoundError("Couldn't find val data loader stored in file")


def generate_val_loader(args, dataset):
    val_dataset = dataset.get_val_dataset()
    X, Y = shuffle_data(val_dataset)

    return dataset.get_data_loader_from_data(args.val_batch_size, X, Y)


def load_test_data_loader(logger, args):
    """
    Loads the test data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if args.by_attack_type:
        if os.path.exists(args.test_data_loader_by_attack_type_pickle_path):
            return load_data_loader_from_file(logger, args.test_data_loader_by_attack_type_pickle_path)
        else:
            logger.error("Couldn't find test data loader by attack type stored in file")

            raise FileNotFoundError(
                "Couldn't find test data loader by attack type stored in file")
    else:
        if os.path.exists(args.test_data_loader_pickle_path):
            return load_data_loader_from_file(logger, args.test_data_loader_pickle_path)
        else:
            logger.error("Couldn't find test data loader stored in file")

            raise FileNotFoundError(
                "Couldn't find test data loader stored in file")


def generate_test_loader(args, dataset):
    test_dataset = dataset.get_test_dataset()
    X, Y = shuffle_data(test_dataset)

    return dataset.get_data_loader_from_data(args.test_batch_size, X, Y)


def load_mal_data_loader(logger, args):
    """
    Loads the mal data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if args.by_attack_type:
        if os.path.exists(args.mal_data_loader_by_attack_type_pickle_path):
            return load_data_loader_from_file(logger, args.mal_data_loader_by_attack_type_pickle_path)
        else:
            logger.error("Couldn't find mal data loader by attack type stored in file")

            raise FileNotFoundError(
                "Couldn't find mal data loader by attack type stored in file")
    else:
        if os.path.exists(args.mal_data_loader_pickle_path):
            return load_data_loader_from_file(logger, args.mal_data_loader_pickle_path)
        else:
            logger.error("Couldn't find mal data loader stored in file")

            raise FileNotFoundError("Couldn't find mal data loader stored in file")


def generate_mal_loader(args, dataset):
    mal_dataset = dataset.get_mal_dataset()
    X, Y = shuffle_data(mal_dataset)

    return dataset.get_data_loader_from_data(args.mal_batch_size, X, Y)


def load_data_loader_from_file(logger, filename):
    """
    Loads DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param filename: string
    """
    logger.info("Loading data loader from file: {}".format(filename))

    with open(filename, "rb") as f:
        return load_saved_data_loader(f)


def shuffle_data(dataset):
    data = list(zip(dataset[0], dataset[1]))
    random.shuffle(data)
    X, Y = zip(*data)
    X = numpy.asarray(X)
    Y = numpy.asarray(Y)

    return X, Y


def load_saved_data_loader(file_obj):
    return pickle.load(file_obj)


def save_data_loader_to_file(data_loader, file_obj):
    pickle.dump(data_loader, file_obj)
