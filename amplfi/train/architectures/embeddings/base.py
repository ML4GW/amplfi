import torch


class Embedding(torch.nn.Module):
    """
    Dummy base class for embedding networks.

    All embeddings should accept `num_ifos`
    as their first argument. They should also
    define a `context_dim` attribute that returns
    the dimensionality of the output of the network,
    which will be used to instantiate the flow transorms.

    This class obvioulsy isn't necessary, but leaving this
    as a reminder that we may wan't to enforce
    the above behavior more explicitly in the future
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_dim = None
