from loguru import logger
import torch
import pathlib
import os
import argparse
from function.arguments import Arguments
from function.datasets import DataReader
from function.nets import AE, VAE, DualLossAE, SupAE
from function.utils import (
    generate_train_loader,
    generate_val_loader,
    generate_test_loader,
    generate_mal_loader,
    save_data_loader_to_file,
)
import yaml
from function.utils.common_util import get_exp_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config Environment")
    parser.add_argument("-data", type=str, default="nb_iot", help="Dataset to use")

    arg_parser = parser.parse_args()
    dataset = arg_parser.data
    # cic_ids, nb_iot, nsl_kdd, nsl_kdd_one_class, unsw, unsw_big, unsw_one_class, spambase, ctu13_08 09 10 13, internet_ad

    with open(f"./config/{dataset}.yaml", "r") as stream:
        data_loaded = yaml.safe_load(stream)
        list_configs = get_exp_config(data_loaded)

        args = Arguments(logger, list_configs[0])
        
        dataset = DataReader(args, args.dataset)
        if not os.path.exists("data_loaders/{}".format(args.dataset)):
            pathlib.Path("data_loaders/{}".format(args.dataset)).mkdir(
                parents=True, exist_ok=True
            )

        train_data_loader = generate_train_loader(args, dataset)
        val_data_loader = generate_val_loader(args, dataset)
        test_data_loader = generate_test_loader(args, dataset)
        mal_data_loader = generate_mal_loader(args, dataset)

        with open(args.train_data_loader_pickle_path, "wb") as f:
            save_data_loader_to_file(train_data_loader, f)

        with open(args.val_data_loader_pickle_path, "wb") as f:
            save_data_loader_to_file(val_data_loader, f)

        with open(args.test_data_loader_pickle_path, "wb") as f:
            save_data_loader_to_file(test_data_loader, f)

        with open(args.mal_data_loader_pickle_path, "wb") as f:
            save_data_loader_to_file(mal_data_loader, f)
        
        # -----------------------------------------------------
        # ----------- Attack type classìication experiment ----------
        # -----------------------------------------------------

        args.by_attack_type = True
        dataset = DataReader(args, args.dataset)
        if not os.path.exists("data_loaders_by_attack_type/{}".format(args.dataset)):
            pathlib.Path("data_loaders_by_attack_type/{}".format(args.dataset)).mkdir(
                parents=True, exist_ok=True
            )

        train_data_loader = generate_train_loader(args, dataset)
        val_data_loader = generate_val_loader(args, dataset)
        test_data_loader = generate_test_loader(args, dataset)
        mal_data_loader = generate_mal_loader(args, dataset)

        with open(args.train_data_loader_by_attack_type_pickle_path, "wb") as f:
            save_data_loader_to_file(train_data_loader, f)

        with open(args.val_data_loader_by_attack_type_pickle_path, "wb") as f:
            save_data_loader_to_file(val_data_loader, f)

        with open(args.test_data_loader_by_attack_type_pickle_path, "wb") as f:
            save_data_loader_to_file(test_data_loader, f)

        with open(args.mal_data_loader_by_attack_type_pickle_path, "wb") as f:
            save_data_loader_to_file(mal_data_loader, f)

        # -------------------------------------------
        # ----------- Model Initialization ----------
        # -------------------------------------------

        if not os.path.exists(args.default_model_folder_path):
            os.mkdir(args.default_model_folder_path)

        args.logger.debug(
            "Initialize the AE model with the dimension of {}".format(args.dimension)
        )
        full_save_path = os.path.join(args.default_model_folder_path, "AE.model")
        torch.save(AE(args.dimension).state_dict(), full_save_path)

        args.logger.debug(
            "Initialize the VAE model with the dimension of {}".format(args.dimension)
        )
        full_save_path = os.path.join(args.default_model_folder_path, "VAE.model")
        torch.save(VAE(args.dimension).state_dict(), full_save_path)

        args.logger.debug(
            "Initialize the DualLossAE model with the dimension of {}".format(args.dimension)
        )
        full_save_path = os.path.join(args.default_model_folder_path, "DualLossAE.model")
        torch.save(DualLossAE(args.dimension).state_dict(), full_save_path)

        args.logger.debug(
            "Initialize the SupAE model with the dimension of {}".format(args.dimension)
        )
        full_save_path = os.path.join(args.default_model_folder_path, "SupAE.model")
        torch.save(SupAE(args.dimension).state_dict(), full_save_path)
        
        args.logger.debug(f"Initialize {dataset} environment successfully.")
