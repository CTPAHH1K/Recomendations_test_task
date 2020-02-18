from pathlib import Path
import pandas as pd


class DataProvider:
    def __init__(self, data_directory: Path) -> None:
        self._data_directory = data_directory

    def get_purchases_train(self) -> pd.DataFrame:
        return pd.read_csv(self._data_directory / "purchases_train.csv")

    def get_purchases_test(self) -> pd.DataFrame:
        return pd.read_csv(self._data_directory / "purchases_test.csv")

    def get_customers(self) -> pd.DataFrame:
        return pd.read_csv(self._data_directory / "customers.csv")
