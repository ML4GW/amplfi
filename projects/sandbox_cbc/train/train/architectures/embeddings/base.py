from typing import Optional

import torch


class Embedding(torch.nn.Module):
    """
    Base class for embedding networks.
    All embeddings should accept
    `num_ifos` as their first argument, 
    `context_dim` as theri second, and 
    `strain_dim` as their third, since the 
    CLI links to this argument from the datamodule
    at initialization time.
    """