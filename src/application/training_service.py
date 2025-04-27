import torch.backends.mps
from torch.utils.data import DataLoader

from src.config import TRAIN_RATIO, VAL_RATIO, BATCH_SIZE
from src.domain.services.data_splitter import DatasetSplitter
from src.domain.services.model_trainer import ModelTrainer


class TrainingService:
    def __init__(
        self, dataset, model_trainer: ModelTrainer, dataset_splitter: DatasetSplitter
    ):
        self.dataset = dataset
        self.model_trainer = model_trainer
        self.dataset_splitter = dataset_splitter

    def run_training(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        batch_size: int = BATCH_SIZE,
    ):
        train_data, val_data, test_data = self.dataset_splitter.split(
            self.dataset, train_ratio, val_ratio
        )
        train_loader = DataLoader(
            train_data, batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
        )
        val_loader = DataLoader(
            val_data, batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
        )

        # TODO: Use test data for evaluation

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.model_trainer.train(train_loader, val_loader, device)
