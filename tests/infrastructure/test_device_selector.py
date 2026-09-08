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

    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_name", return_value="Mock CUDA Device")
    def test_get_device_info_includes_cuda_details_when_available(
        self,
        mock_device_name: MagicMock,
        mock_count: MagicMock,
        mock_cuda: MagicMock,
        mock_mps: MagicMock,
    ) -> None:
        info = DeviceSelector.get_device_info()
        assert info == {
            "mps_available": "False",
            "cuda_available": "True",
            "cuda_device_count": "2",
            "cuda_device_name": "Mock CUDA Device",
        }

    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_get_device_info_omits_cuda_details_without_cuda(
        self, mock_cuda: MagicMock, mock_mps: MagicMock
    ) -> None:
        info = DeviceSelector.get_device_info()
        assert info == {"mps_available": "True", "cuda_available": "False"}
