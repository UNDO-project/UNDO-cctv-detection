from unittest.mock import MagicMock, patch
from src.infrastructure.device_selector import DeviceSelector


class TestDeviceSelector:
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_selects_mps_when_available(self, mock_mps: MagicMock) -> None:
        device = DeviceSelector.get_optimal_device()
        assert device.type == "mps"

    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="Mock CUDA Device")
    def test_selects_cuda_when_mps_unavailable(
        self, mock_device_name: MagicMock, mock_cuda: MagicMock, mock_mps: MagicMock
    ) -> None:
        device = DeviceSelector.get_optimal_device()
        assert device.type == "cuda"

    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=False)
    def test_selects_cpu_when_no_gpu(
        self, mock_cuda: MagicMock, mock_mps: MagicMock
    ) -> None:
        device = DeviceSelector.get_optimal_device()
        assert device.type == "cpu"

    def test_get_device_info_returns_dict(self) -> None:
        info = DeviceSelector.get_device_info()
        assert isinstance(info, dict)
        assert "mps_available" in info
        assert "cuda_available" in info
