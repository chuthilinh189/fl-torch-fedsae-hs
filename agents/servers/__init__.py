def get_server_class(model_type):
    model_map = {
        "AE": "serverAE",
        "DAE": "serverDAE",
        "SAE": "serverSAE",
        "FedMSE": "serverFedMSE",
        "SAE1": "serverSAE1",
        "SDAE": "serverSDAE",
        "SDAE1": "serverSDAE1",
        "SupAE": "serverSupAE",
        "DualLossAE": "serverDualLossAE"
    }

    if model_type not in model_map:
        raise ValueError(f"Unsupported model type for server: {model_type}")

    module_name = model_map[model_type]
    class_name = "Server" + model_type

    module = __import__(f"agents.servers.{module_name}", fromlist=[class_name])
    ServerClass = getattr(module, class_name)

    return ServerClass
