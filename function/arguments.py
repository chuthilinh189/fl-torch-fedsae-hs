from .nets import AE, VAE, DualLossAE, SupAE, FedDetect
import torch

# Set seed cố định cho reproducibility
SEED = 19
torch.manual_seed(SEED)


class Arguments:
    def __init__(self, logger, config):
        self.logger = logger

        # 🛠 Các tham số nạp từ config (dòng lệnh)
        self.dataset = config["dataset"]
        self.train_batch_size = config["train_batch_size"]
        self.val_batch_size = config["val_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.mal_batch_size = config["mal_batch_size"]
        self.dimension = config["dimension"]
        self.epochs = config["epochs"]
        self.model_type = config["model_type"]
        self.noise_type = config["noise_type"]
        self.num_poisoned_workers = config["num_of_poisoned_workers"]
        self.poisoned_sample_ratio = config["poisoned_sample_ratio"]
        self.learning_rate = config["learning_rate"]
        self.std_noise = config["noise_std"]
        self.attack_std_noise = config["attack_noise_std"]
        self.aggregation_type = config["aggregation_type"]
        self.coef_shrink_ae = config["coef_shrink_ae"]
        self.threshold_multiplier = config["threshold_multiplier"]
        self.num_multi_class_clients = config["num_multi_class_clients"]
        self.by_attack_type = config["by_attack_type"]
        # ---- Partition strategy for client data distribution ----
        # 'original' (default) -> existing behavior (batch-wise round robin)
        # 'hybrid' -> deterministic class-assignment + Dirichlet splitting inside assigned classes
        # If an overall experiment type is provided, adapt some defaults
        # experiment_type can be: 'normal' | 'by_attack_type' | 'non_iid_dir'
        self.experiment_type = config.get("experiment_type", "normal")

        # For non_iid_dir experiments we want the hybrid partitioning
        default_partition = "hybrid" if self.experiment_type == "non_iid_dir" else "original"
        self.partition_strategy = config.get("partition_strategy", default_partition)
        # Number of classes (including normal class 0) each client should see when using hybrid
        self.seen_per_client = config.get("seen_per_client", 5)
        # Dirichlet alpha used for splitting samples of each class across clients
        self.dir_alpha = config.get("dir_alpha", 0.7)
        # Seed used for deterministic assignment of seen classes
        self.assign_seed = config.get("assign_seed", 0)
            # ----- ptl (Prototype-Triplet) hyperparameters -----
        # Lambda weight for prototype-triplet loss term
        self.ptl_lambda = config.get("ptl_lambda", 0.3)
        # EMA coefficient for server-side prototype updates (0..1)
        self.ptl_proto_ema = config.get("ptl_proto_ema", 0.99)
        # Margin used in triplet-style loss (d_pos - d_neg + margin)
        self.ptl_margin = config.get("ptl_margin", 5.0)
        # Distance metric for prototypes: 'euclid' or 'cosine'
        self.ptl_distance = config.get("ptl_distance", "euclid")
        # Decision mode for PTL at test time: 're' | 'proto' | 'combined'
        # default to 'proto' to use prototype-based thresholding
        self.ptl_decision_mode = config.get("ptl_decision_mode", "proto")
        # DualLossAE scaling factor for anomaly term
        self.dual_loss_alpha = config.get("dual_loss_alpha", 1)

        # ⚙️ Các tham số nội bộ tự động gán
        self.num_workers = 20  # Tổng số client mặc định
        self.es_offset = 500  # Offset cho early stopping
        self.threshold_factor = 0  # Threshold scaling
        self.cuda = True  # Luôn bật GPU
        self.shuffle = False  # Không shuffle dữ liệu
        self.loss_function = torch.nn.MSELoss  # Hàm loss mặc định (MSELoss)

        # 📂 Các đường dẫn mặc định
        self.train_data_loader_pickle_path = (
            f"data_loaders/{self.dataset}/train_data_loader.pickle"
        )
        self.val_data_loader_pickle_path = (
            f"data_loaders/{self.dataset}/val_data_loader.pickle"
        )
        self.test_data_loader_pickle_path = (
            f"data_loaders/{self.dataset}/test_data_loader.pickle"
        )
        self.mal_data_loader_pickle_path = (
            f"data_loaders/{self.dataset}/mal_data_loader.pickle"
        )
        self.train_data_loader_by_attack_type_pickle_path = (
            f"data_loaders_by_attack_type/{self.dataset}/train_data_loader.pickle"
        )
        self.val_data_loader_by_attack_type_pickle_path = (
            f"data_loaders_by_attack_type/{self.dataset}/val_data_loader.pickle"
        )
        self.mal_data_loader_by_attack_type_pickle_path = (
            f"data_loaders_by_attack_type/{self.dataset}/mal_data_loader.pickle"
        )
        self.test_data_loader_by_attack_type_pickle_path = (
            f"data_loaders_by_attack_type/{self.dataset}/test_data_loader.pickle"
        )
        self.train_data_loader_full_attack_type_pickle_path = (
            f"data_loaders_full_attack_type/{self.dataset}/train_data_loader.pickle"
        )
        self.val_data_loader_full_attack_type_pickle_path = (
            f"data_loaders_full_attack_type/{self.dataset}/val_data_loader.pickle"
        )
        self.test_data_loader_full_attack_type_pickle_path = (
            f"data_loaders_full_attack_type/{self.dataset}/test_data_loader.pickle"
        )
        self.mal_data_loader_full_attack_type_pickle_path = (
            f"data_loaders_full_attack_type/{self.dataset}/mal_data_loader.pickle"
        )
        self.default_model_folder_path = f"default_models/{self.dataset}"

        # If experiment_type requests full-attack-type data, point defaults to those pickles
        if self.experiment_type == "non_iid_dir":
            self.train_data_loader_pickle_path = self.train_data_loader_full_attack_type_pickle_path
            self.val_data_loader_pickle_path = self.val_data_loader_full_attack_type_pickle_path
            self.test_data_loader_pickle_path = self.test_data_loader_full_attack_type_pickle_path
            self.mal_data_loader_pickle_path = self.mal_data_loader_full_attack_type_pickle_path

        # 🛠 Tạo biến để lưu log train/test sau này
        self.train_log_df = None
        self.test_log_df = None

        # Xử lý replacement ratio nếu có poisoned workers
        if self.num_poisoned_workers == 0:
            self.replacement_ratio = 0
        else:
            self.replacement_ratio = self.poisoned_sample_ratio

    def get_net(self, model_type):
        """Trả về kiến trúc mạng tương ứng"""
        if model_type == "VAE":
            return VAE
        elif model_type == "DualLossAE":
            return DualLossAE
        elif model_type == "SupAE":
            return SupAE
        elif model_type in ("FedHome", "FedGH"):
            return FedDetect
        return AE

    def set_train_log_df(self, df):
        self.train_log_df = df

    def get_train_log_df(self):
        return self.train_log_df

    def set_test_log_df(self, df):
        self.test_log_df = df

    def get_test_log_df(self):
        return self.test_log_df

    def log(self):
        """In ra các tham số để debug."""
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        """Trả về chuỗi mô tả toàn bộ tham số."""
        return (
            "\n[Experiment Config]"
            + "\nModel Type: {}".format(self.model_type)
            + "\nNoise Type: {}".format(self.noise_type)
            + "\nAggregation Type: {}".format(self.aggregation_type)
            + "\nLearning Rate: {}".format(self.learning_rate)
            + "\nCoefficient Shrink AE: {}".format(self.coef_shrink_ae)
            + "\nThreshold Multiplier: {}".format(self.threshold_multiplier)
            + "\nDual Loss Alpha: {}".format(self.dual_loss_alpha)
            + "\nNoise STD: {}".format(self.std_noise)
            + "\nAttack Noise STD: {}".format(self.attack_std_noise)
            + "\nNumber of Poisoned Clients: {}".format(self.num_poisoned_workers)
            + "\nPoisoned Sample Ratio: {}".format(self.poisoned_sample_ratio)
            + "\nDimension: {}".format(self.dimension)
            + "\nEpochs: {}".format(self.epochs)
            + "\nNumber of Multi-Class Clients: {}".format(self.num_multi_class_clients)
            + "\nptl Lambda: {}".format(self.ptl_lambda)
            + "\nptl Prototype EMA: {}".format(self.ptl_proto_ema)
            + "\nptl Margin: {}".format(self.ptl_margin)
            + "\nptl Distance: {}".format(self.ptl_distance)
            + "\nptl Decision Mode: {}".format(self.ptl_decision_mode)
            + "\n\n[System Config]"
            + "\nNumber of Workers: {}".format(self.num_workers)
            + "\nEarly Stop Offset: {}".format(self.es_offset)
            + "\nThreshold Factor: {}".format(self.threshold_factor)
            + "\nCuda Enabled: {}".format(self.cuda)
            + "\nShuffle: {}".format(self.shuffle)
            + "\nLoss Function: {}".format(self.loss_function)
            + "\n\n[Dataset Config]"
            + "\nDataset: {}".format(self.dataset)
            + "\nTrain Batch Size: {}".format(self.train_batch_size)
            + "\nVal Batch Size: {}".format(self.val_batch_size)
            + "\nTest Batch Size: {}".format(self.test_batch_size)
            + "\nMalicious Batch Size: {}".format(self.mal_batch_size)
            + "\nTrain Data Path: {}".format(self.train_data_loader_pickle_path)
            + "\nVal Data Path: {}".format(self.val_data_loader_pickle_path)
            + "\nTest Data Path: {}".format(self.test_data_loader_pickle_path)
            + "\nMalicious Data Path: {}".format(self.mal_data_loader_pickle_path)
            + "\nDefault Model Save Path: {}".format(self.default_model_folder_path)
            + "\nPartition Strategy: {}".format(self.partition_strategy)
            + "\nSeen per client (hybrid): {}".format(self.seen_per_client)
            + "\nDirichlet alpha (hybrid): {}".format(self.dir_alpha)
            + "\nAssign seed (hybrid): {}".format(self.assign_seed)
            + "\n"
        )
