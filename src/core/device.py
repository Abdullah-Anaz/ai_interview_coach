import torch


def get_compute_device() -> torch.device:
    """
    Determines and returns the optimal available hardware compute device.

    Returns:
        torch.device: A PyTorch device object pointing to 'cuda' if a compatible 
        NVIDIA GPU is available, otherwise falling back to 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")