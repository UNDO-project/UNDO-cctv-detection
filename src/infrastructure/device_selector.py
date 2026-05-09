import torch
from loguru import logger


class DeviceSelector:
    """Selects the best available device for PyTorch operations.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    """

    @staticmethod
    def get_optimal_device() -> torch.device:
        """Select best available device.

        Example::

            device = DeviceSelector.get_optimal_device()
            model.to(device)

        :return: The optimal device for training/inference.
        :rtype: torch.device
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {device_name}")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU available, using CPU (training will be slow)")

        return device

    @staticmethod
    def get_device_info() -> dict[str, str]:
        """
        Get information about available devices.
        :return: Dict with device information for debugging/logging.
        """
        info = {
            "mps_available": str(torch.backends.mps.is_available()),
            "cuda_available": str(torch.cuda.is_available()),
        }

        if torch.cuda.is_available():
            info["cuda_device_count"] = str(torch.cuda.device_count())
            info["cuda_device_name"] = torch.cuda.get_device_name(0)

        return info
