from function.utils import (
    distribute_batches_equally,
    convert_distributed_data_into_numpy,
    poison_data,
    generate_data_loaders_from_distributed_dataset,
    filter_data_loader_by_label
)

def client_data_process(args, data_loader, poisoned_workers, mal_data_loader, batch_size, poison=True):
    distributed_dataset = distribute_batches_equally(data_loader, args.num_workers)
    distributed_dataset = convert_distributed_data_into_numpy(distributed_dataset)

    if poison:
        distributed_dataset = poison_data(
            args.logger,
            distributed_dataset,
            args.num_workers,
            poisoned_workers,
            args.noise_type,
            mal_data_loader,
            args.replacement_ratio,
            args.attack_std_noise,
        )

    data_loaders = generate_data_loaders_from_distributed_dataset(distributed_dataset, batch_size)
    return data_loaders
