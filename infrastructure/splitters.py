from sklearn.model_selection import train_test_split

from domain.services.data_splitter import DataSplitter


class SklearnDatasetSplitter(DataSplitter):
    def split(self, dataset, train_ratio: float, val_ratio: float):
        train_data, temp_data = train_test_split(
            dataset, train_size=train_ratio, random_state=42
        )

        # Compute the validation size as a fraction of the remaining data.
        val_size = val_ratio / (1 - train_ratio)
        val_data, test_data = train_test_split(
            temp_data, train_size=val_size, random_state=42
        )
        return train_data, val_data, test_data
