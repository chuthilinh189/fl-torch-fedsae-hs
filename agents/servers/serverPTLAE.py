from agents.servers.serverPTL import ServerPTL

class ServerPTLAE(ServerPTL):
    """Thin wrapper to allow model_type 'PTLAE' to reuse ServerPTL logic."""
    def __init__(self, args):
        super().__init__(args)
