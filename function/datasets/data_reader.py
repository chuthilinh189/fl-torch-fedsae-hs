from .dataset import Dataset
from .data_load import *


class DataReader(Dataset):
    def __init__(self, args, data_name):
        self.data_name = data_name
        self.args = args
        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            self.X_mal,
            self.y_mal,
        ) = self.load_dataset()
        super(DataReader, self).__init__(args)

    def load_dataset(self):
        if self.data_name == "bot_iot":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = bot_iot()
        if self.data_name == "cic_ids":
            if getattr(self.args, "full_attack_type", False):
                X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = cic_ids_full_attack_type()
            elif self.args.by_attack_type:
                X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = cic_ids_by_attack_type()
            else:
                X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = cic_ids()
        if self.data_name == "cic_iot":
            print("Loading CIC IoT dataset")
            if getattr(self.args, "full_attack_type", False):
                X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = cic_iot_full_attack_type()
            else:
                X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = cic_iot()
        if self.data_name == "nb_iot":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = nb_iot()
        if self.data_name == "nsl_kdd":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = nsl_kdd()
        if self.data_name == "nsl_kdd_one_class":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = (
                nsl_kdd_one_class()
            )
        if self.data_name == "unsw":
            if getattr(self.args, "full_attack_type", False):
                X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = unsw_full_attack_type()
            elif self.args.by_attack_type:
                X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = unsw_by_attack_type()
            else:
                X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = unsw()
        if self.data_name == "unsw_big":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = unsw_big()
        if self.data_name == "unsw_one_class":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = (
                unsw_one_class()
            )
        if self.data_name == "spambase":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = spambase()
        if self.data_name == "ctu13_08":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = ctu13_08()
        if self.data_name == "ctu13_09":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = ctu13_09()
        if self.data_name == "ctu13_10":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = ctu13_10()
        if self.data_name == "ctu13_13":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = ctu13_13()
        if self.data_name == "internet_ad":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = internet_ad()
        if self.data_name == "wsn_ds":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = wsn_ds()
        if self.data_name == "ton_iot_fridge":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = ton_iot_fridge()
        if self.data_name == "ton_iot_weather":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = ton_iot_weather()
        if self.data_name == "ton_iot_modbus":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = ton_iot_modbus()
        if self.data_name == "ton_iot_network":
            X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal = ton_iot_network()


        return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

    def load_train_dataset(self):
        self.args.logger.debug("Loading {} train data".format(self.data_name))

        train_loader = self.get_data_loader_from_data(
            len(self.X_train), self.X_train, self.y_train
        )
        train_data = self.get_tuple_from_data_loader(train_loader)

        self.args.logger.debug("Finished loading {} train data".format(self.data_name))
        return train_data

    def load_val_dataset(self):
        self.args.logger.debug("Loading {} val data".format(self.data_name))

        val_loader = self.get_data_loader_from_data(
            len(self.X_val), self.X_val, self.y_val
        )
        val_data = self.get_tuple_from_data_loader(val_loader)

        self.args.logger.debug("Finished loading {} val data".format(self.data_name))
        return val_data

    def load_test_dataset(self):
        self.args.logger.debug("Loading {} test data".format(self.data_name))

        test_loader = self.get_data_loader_from_data(
            len(self.X_test), self.X_test, self.y_test
        )
        test_data = self.get_tuple_from_data_loader(test_loader)

        self.args.logger.debug("Finished loading {} test data".format(self.data_name))
        return test_data

    def load_mal_dataset(self):
        self.args.logger.debug("Loading {} mal data".format(self.data_name))

        mal_loader = self.get_data_loader_from_data(
            len(self.X_mal), self.X_mal, self.y_mal
        )
        mal_data = self.get_tuple_from_data_loader(mal_loader)

        self.args.logger.debug("Finished loading {} mal data".format(self.data_name))
        return mal_data
