def run_machine_learning(server, clients, poisoned_workers, args):
    """Run training loop. PdfPages visualization removed (not used).

    Calls server.train_on_clients(epoch, clients, poisoned_workers) without pdf_writer.
    """
    for epoch in range(1, args.epochs + 1):
        no_client_training = server.train_on_clients(epoch, clients, poisoned_workers)

        if no_client_training:
            break

    server.test_on_clients(epoch, clients, poisoned_workers)
