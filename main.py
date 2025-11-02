from core.experiment_runner import run_exp
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config Environment")

    parser.add_argument("-d", type=str, required=True, help="Dataset name (e.g., cic_ids, nslkdd)")
    parser.add_argument("-m", type=str, required=True, help="Model type (AE, DAE, SAE, SDAE, SupAE, DualLossAE)")
    parser.add_argument("-tbs", type=int, default=128, help="Train batch size")
    parser.add_argument("-vbs", type=int, default=128, help="Val batch size")
    parser.add_argument("-di", type=int, default=128, help="Dimension")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-ep", type=int, default=4000, help="Epochs")
    parser.add_argument("-agg", type=str, default="average", help="Aggregation type")
    parser.add_argument("-nt", type=str, default="label_flipping", help="Noise type")
    parser.add_argument("-pw", type=int, default=0, help="Number of poisoned clients")
    parser.add_argument("-pr", type=float, default=1.0, help="Poisoned sample ratio")
    parser.add_argument("-ns", type=float, default=0.001, help="Noise stddev")
    parser.add_argument("-ans", type=float, default=3.0, help="Attack noise stddev")
    parser.add_argument("-cs", type=float, default=1.0, help="Coefficient shrink AE")
    parser.add_argument("-tm", type=float, default=0.0, help="Threshold multiplier")
    parser.add_argument("-mc", type=int, default=0, help="Multi-class client count")
    parser.add_argument("-at", type=bool, default=False, help="By attack type (False/ True)")

    args = parser.parse_args()

    config = {
        "dataset": args.d,
        "train_batch_size": args.tbs,
        "val_batch_size": args.vbs,
        "test_batch_size": 1,
        "mal_batch_size": args.tbs,
        "dimension": args.di,
        "epochs": args.ep,
        "model_type": args.m,
        "noise_type": args.nt,
        "num_of_poisoned_workers": args.pw,
        "poisoned_sample_ratio": args.pr,
        "learning_rate": args.lr,
        "noise_std": args.ns,
        "attack_noise_std": args.ans,
        "aggregation_type": args.agg,
        "coef_shrink_ae": args.cs,
        "threshold_multiplier": args.tm,
        "num_multi_class_clients": args.mc,
        "by_attack_type": args.at,
    }

    run_exp(config)
