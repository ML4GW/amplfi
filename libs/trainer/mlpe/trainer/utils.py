import torch


def number_trainable_parameters(model: torch.nn.Module):
    """Return a list of trainable parameters in a model"""
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num += sum(p.numel() for p in model.buffers() if p.requires_grad)
    return num


def model_memory(model: torch.nn.Module):
    """Return the memory used by a model in Mega bytes"""
    memory = sum(p.numel() * p.element_size() for p in model.parameters())
    memory += sum(p.numel() * p.element_size() for p in model.buffers())
    return memory / 1e6
