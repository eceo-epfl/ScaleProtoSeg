"""
Pytorch Lightning DataModule for training prototype segmentation model on Cityscapes, PASCAL, ADE20K, COCO-Stuff, and EM datasets
"""

import multiprocessing
import os

import gin
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from segmentation.data.dataset import PatchClassificationDataset
from settings import data_path


@gin.configurable(denylist=["batch_size"])
class PatchClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_type: str = gin.REQUIRED,
        dataloader_n_jobs: int = gin.REQUIRED,
        train_key: str = "train",
    ):
        super().__init__()
        self.dataloader_n_jobs = dataloader_n_jobs if dataloader_n_jobs != -1 else multiprocessing.cpu_count()
        self.batch_size = batch_size
        self.train_key = train_key
        self.data_type = data_type

    def prepare_data(self):
        if not os.path.exists(os.path.join(data_path[self.data_type], "annotations")):
            raise ValueError("Please download dataset and preprocess it using the preprocess scripts")

    def get_data_loader(self, dataset: PatchClassificationDataset, **kwargs) -> DataLoader:
        """Create a DataLoader for the given dataset

        Args:
            dataset (PatchClassificationDataset): Dataset to create DataLoader for

        Returns:
            DataLoader: DataLoader for the given dataset
        """

        if "batch_size" in kwargs:
            return DataLoader(
                dataset=dataset, shuffle=not dataset.is_eval, num_workers=self.dataloader_n_jobs, **kwargs
            )
        return DataLoader(
            dataset=dataset,
            shuffle=not dataset.is_eval,
            num_workers=self.dataloader_n_jobs,
            batch_size=self.batch_size,
            **kwargs,
        )

    def train_dataloader(self, **kwargs):
        train_split = PatchClassificationDataset(
            split_key=self.train_key,
            is_eval=False,
        )
        return self.get_data_loader(train_split, **kwargs)

    def val_dataloader(self, **kwargs):
        val_split = PatchClassificationDataset(
            split_key="val",
            is_eval=True,
        )
        return self.get_data_loader(val_split, **kwargs)

    def test_dataloader(self, **kwargs):
        test_split = PatchClassificationDataset(
            split_key="val",  # We do not use PASCAL & Cityscapes test set
            is_eval=True,
        )
        return self.get_data_loader(test_split, **kwargs)

    def train_push_dataloader(self, **kwargs):
        train_split = PatchClassificationDataset(split_key="train", is_eval=True, push_prototypes=True)
        return self.get_data_loader(train_split, **kwargs)
