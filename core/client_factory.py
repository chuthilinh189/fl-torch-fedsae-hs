from agents.clients import get_client_class

def create_clients(args, train_loaders, val_loaders, test_loaders):
    clients = []
    ClientClass = get_client_class(args.model_type)

    for idx in range(args.num_workers):
        clients.append(
            ClientClass(
                args,
                idx,
                train_loaders[idx],
                val_loaders[idx],
                test_loaders[idx],
            )
        )
    return clients
