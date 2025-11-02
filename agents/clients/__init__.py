def get_client_class(model_type):
    model_map = {
        "AE": "clientAE",
        "DAE": "clientDAE",
        "SAE": "clientSAE",
        "FedMSE": "clientFedMSE",
        "SAE1": "clientSAE1",
        "SDAE": "clientSDAE",
        "SDAE1": "clientSDAE1",
        "SupAE": "clientSupAE",
        "DualLossAE": "clientDualLossAE"
    }

    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    module_name = model_map[model_type]
    class_name = "Client" + model_type

    # Import module động
    module = __import__(f"agents.clients.{module_name}", fromlist=[class_name])
    ClientClass = getattr(module, class_name)

    return ClientClass
