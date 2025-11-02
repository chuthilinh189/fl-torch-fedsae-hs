from matplotlib.backends.backend_pdf import PdfPages

def run_machine_learning(server, clients, poisoned_workers, args):
    pdf_path = f"logs/{args.model_type}_visualize_params.pdf"
    pdf_writer = PdfPages(pdf_path)

    for epoch in range(1, args.epochs + 1):
        no_client_training = server.train_on_clients(epoch, clients, poisoned_workers, pdf_writer)
        
        if no_client_training:
            break

    pdf_writer.close()
    server.test_on_clients(epoch, clients, poisoned_workers)
