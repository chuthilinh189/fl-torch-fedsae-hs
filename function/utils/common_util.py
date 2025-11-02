import random
import itertools


def get_exp_config(config):
    print("================ Read Config YAML =================")
    print(config)

    list_configs = []
    list_arrays = []
    for key in config:
        if type(config[key]) == list:
            list_arrays.append(config[key])
    config_combinations = list(itertools.product(*list_arrays))

    for combination in config_combinations:
        new_config = {
            "dataset": config["dataset"],
            "train_batch_size": config["train_batch_size"],
            "val_batch_size": config["val_batch_size"],
            "test_batch_size": config["test_batch_size"],
            "mal_batch_size": config["mal_batch_size"],
            "dimension": config["dimension"],
            "epochs": config["epochs"],
            "model_type": combination[0],
            "noise_type": combination[1],
            "num_of_poisoned_workers": combination[2],
            "poisoned_sample_ratio": combination[3],
            "learning_rate": combination[4],
            "noise_std": combination[5],
            "attack_noise_std": combination[6],
            "aggregation_type": combination[7],
            "coef_shrink_ae": combination[8],
            "threshold_multiplier": combination[9],
            "num_multi_class_clients": combination[10],
            "by_attack_type": combination[11],
        }
        list_configs.append(new_config)
    print(
        f"================ Num of EXP: {len(list_configs)} =================")
    return list_configs


def identify_random_elements(max, num_random_elements):
    """
    Picks a specified number of random elements from 0 - max.

    :param max: Max number to pick from
    :type max: int
    :param num_random_elements: Number of random elements to select
    :type num_random_elements: int
    :return: list
    """
    if num_random_elements > max:
        return []

    ids = []
    x = 0
    while x < num_random_elements:
        rand_int = random.randint(0, max - 1)

        if rand_int not in ids:
            ids.append(rand_int)
            x += 1
    if num_random_elements == 0:
        return []
    return ids
    return [
        135,
        176,
        171,
        72,
        113,
        102,
        29,
        21,
        39,
        193,
        5,
        125,
        16,
        77,
        84,
        52,
        151,
        79,
        114,
        15,
        107,
        143,
        40,
        20,
        158,
        195,
        137,
        54,
        144,
        139,
        86,
        108,
        68,
        80,
        161,
        148,
        14,
        146,
        43,
        194,
        153,
        181,
        93,
        130,
        115,
        140,
        185,
        35,
        88,
        97,
        71,
        186,
        122,
        188,
        18,
        8,
        154,
        182,
        109,
        134,
        133,
        149,
        64,
        62,
        156,
        101,
        106,
        98,
        83,
        46,
        104,
        172,
        191,
        56,
        119,
        96,
        169,
        132,
        99,
        19,
    ]
    # return ids
