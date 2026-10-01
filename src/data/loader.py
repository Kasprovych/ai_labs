"""Data loading module adhering to Single Responsibility Principle (SRP)
and Interface Segregation Principle (ISP).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import gzip
import shutil
import ssl

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = None

import pandas as pd

from src.config.app_config import DataConfig, PathConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetContainer:
    """Immutable Data Transfer Object (DTO) holding raw dataframe and metadata."""
    data: pd.DataFrame
    target_column: str

    @property
    def features(self) -> pd.DataFrame:
        return self.data.drop(columns=[self.target_column])

    @property
    def target(self) -> pd.Series:
        return self.data[self.target_column]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape


class IDataLoader(ABC):
    """Abstract interface defining the contract for data loading."""

    @abstractmethod
    def load(self) -> DatasetContainer:
        """Load and return the dataset container."""
        pass


class CreditCardDataLoader(IDataLoader):
    """DataLoader implementation for Credit Card Fraud Detection dataset.

    Follows DIP (Dependency Inversion) by receiving configurations via constructor injection.
    """

    OPENML_DATASET_ID = 42175
    # Reliable direct mirror for the Kaggle creditcard.csv dataset
    MIRROR_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"

    def __init__(self, data_config: DataConfig, path_config: PathConfig) -> None:
        self._data_config = data_config
        self._path_config = path_config
        self._csv_path = self._path_config.data_dir / "creditcard.csv"

    def load(self) -> DatasetContainer:
        """Loads data from local cache, direct URL mirror, or OpenML; raises if none is available."""
        df: Optional[pd.DataFrame] = None

        # 1. Check local cache
        if self._csv_path.exists():
            logger.info("Found cached dataset at %s", self._csv_path)
            try:
                df = pd.read_csv(self._csv_path)
            except Exception as e:
                logger.warning("Failed reading cached CSV: %s. Re-fetching.", e)

        # 2. If not local, try downloading from mirror URL
        if df is None:
            logger.info("Local file not found. Fetching from mirror: %s", self.MIRROR_URL)
            try:
                self._path_config.data_dir.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(self.MIRROR_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, context=SSL_CONTEXT) as response, open(self._csv_path, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
                df = pd.read_csv(self._csv_path)
                logger.info("Successfully downloaded and cached dataset (%d rows)", len(df))
            except Exception as e:
                logger.warning("Mirror download failed: %s. Trying sklearn fetch_openml...", e)

        # 3. If mirror failed, try sklearn openml
        if df is None:
            try:
                from sklearn.datasets import fetch_openml
                logger.info("Fetching CreditCardFraudDetection via OpenML (ID: %d)...", self.OPENML_DATASET_ID)
                bunch = fetch_openml(data_id=self.OPENML_DATASET_ID, as_frame=True, parser="auto")
                df = bunch.frame
                # Normalize target name
                if "Class" not in df.columns and "class" in df.columns:
                    df.rename(columns={"class": "Class"}, inplace=True)
                df["Class"] = df["Class"].astype(int)
                df.to_csv(self._csv_path, index=False)
                logger.info("Saved OpenML dataset to %s", self._csv_path)
            except Exception as e:
                logger.warning("OpenML fetch failed: %s", e)

        # 4. No source available: fail loudly rather than train on fabricated data
        if df is None:
            raise RuntimeError(
                "Dataset not available: place Kaggle creditcard.csv into "
                f"{self._csv_path} or check network access to the mirror / OpenML."
            )

        # Ensure correct target data type (int 0 and 1)
        df[self._data_config.target_column] = df[self._data_config.target_column].astype(int)

        logger.info(
            "Dataset ready: %d records, %d features. Class balance: 0=%d, 1=%d (%.3f%% fraud)",
            len(df),
            len(df.columns) - 1,
            (df[self._data_config.target_column] == 0).sum(),
            (df[self._data_config.target_column] == 1).sum(),
            (df[self._data_config.target_column] == 1).mean() * 100
        )

        return DatasetContainer(data=df, target_column=self._data_config.target_column)
