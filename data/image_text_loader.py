from pathlib import Path
from random import randint, choice
import json

import pandas as pd
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class TextImageDataset(Dataset):
    def __init__(self,
                 lmdb_patches_path: str,
                 text_embeddings_path: str,
                 wsi_text: dict,
                 crossvalidation_data: list,
                 shuffle: bool = True,
                 training_phase: bool = True,
                 validation_labels= None,
                 context_length: int = 32
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
        """
        super().__init__()
        self.lmdb_patches_path = lmdb_patches_path
        self.text_embeddings_path = text_embeddings_path
        self.listed_data = crossvalidation_data
        self.shuffle = shuffle
        self.training_phase = training_phase
        self.validation_labels = validation_labels
        self.wsi_text = wsi_text

        self.context_length = context_length

        print("Number of Images: ", len(self.listed_data))        

    def __len__(self):
        return len(self.listed_data)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind: int):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind: int):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind: int):

        key = self.listed_data[ind]

        visual_embeddings = torch.load(str(Path(self.lmdb_patches_path) / f"{Path(key)}"))

        # For now, we are taking the mean, this too can be learned
        visual_embeddings = visual_embeddings.mean(dim=0)
        
        text_path = os.path.join(self.text_embeddings_path, key.split(".pt")[0]+".txt_latent.pt")
        if Path(text_path).exists():
            embedded_sequence = torch.load(text_path)
            next_sequence = torch.tensor(self.wsi_text[key.split(".pt")[0]+".txt"])

        else:
            # it seems that some texts are given the wrong extension
            # some texts are named .svs while they should be .mrxs
            # TODO: Fix it properly 
            text_path = os.path.join(self.text_embeddings_path, key.split(".")[0]+".mrxs.txt_latent.pt")
            
            if Path(text_path).exists():
                embedded_sequence = torch.load(text_path)[:self.context_length+1]
                next_sequence = torch.tensor(self.wsi_text[key.split(".")[0]+".mrxs.txt"])
                
            else:
                return self.skip_sample(ind)

        token_sequence = next_sequence[1:self.context_length+1]
        embedded_sequence = embedded_sequence[:self.context_length]
        #embedded_sequence = next_sequence[:self.context_length]

        return key, visual_embeddings, embedded_sequence, token_sequence

class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 lmdb_patches_path: str,
                 text_embeddings_path: str,
                 text_tokens_path: str,
                 crossvalidation_path: str,
                 batch_size: int,
                 dataset: str,
                 num_workers: int = 0,
                 shuffle: bool = True,
                 val_fold: int = 9,
                 context_length: int = 32
                 ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
        """
        super().__init__()
        self.lmdb_patches_path = lmdb_patches_path
        self.listed_data_path = crossvalidation_path
        self.text_embeddings_path = text_embeddings_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.val_fold = val_fold
        self.dataset = dataset
        self.context_length = context_length

        with open(text_tokens_path, "r") as f:
            self.wsi_tokens = json.load(f)

        validation_folds = [self.val_fold]
        train_folds = list(set(range(0,10)) - set(validation_folds))

        lmdb_patches = os.listdir(lmdb_patches_path)
        lmdb_patches = {lmdb_patch.split(".pt")[0]: lmdb_patch for lmdb_patch in lmdb_patches}

        listed_data_df = pd.read_csv(self.listed_data_path)
        
        training_folds = listed_data_df[listed_data_df["Fold"].isin(train_folds)]
        validation_folds = listed_data_df[listed_data_df["Fold"].isin(validation_folds)]

        # gives us the ability to choose on which dataset to train
        if self.dataset=="radboud":
            training_folds = training_folds[training_folds["WSI"].str.contains("EX")]
            validation_folds = validation_folds[validation_folds["WSI"].str.contains("EX")]

        elif self.dataset=="catania":
            training_folds = training_folds[~training_folds["WSI"].str.contains("EX")]
            validation_folds = validation_folds[~validation_folds["WSI"].str.contains("EX")]

        self.listed_train_data = training_folds.WSI.tolist()
        self.listed_train_data = [lmdb_patches[wsi] for wsi in self.listed_train_data if wsi in lmdb_patches]

        self.listed_val_data = validation_folds.WSI.tolist()
        self.listed_val_data = [lmdb_patches[wsi] for wsi in self.listed_val_data if wsi in lmdb_patches]
        self.listed_val_labels = [wsi_labels[1:-2].astype(np.int) for wsi_labels in validation_folds.to_numpy() if wsi_labels[0] in lmdb_patches]

        print("train_data length: ", len(self.listed_train_data))
        print("val_data length: ", len(self.listed_val_data))
        print("validation labels number: ", len(self.listed_val_labels))
    
    def setup(self, stage: str):

        print("num workers: ", self.num_workers)
        self.train_dataset = TextImageDataset(self.lmdb_patches_path,
                                              self.text_embeddings_path,
                                              self.wsi_tokens,
                                              self.listed_train_data,
                                              shuffle=self.shuffle,
                                              training_phase=True, context_length=self.context_length)

        self.val_dataset = TextImageDataset(self.lmdb_patches_path,
                                              self.text_embeddings_path,
                                              self.wsi_tokens,
                                              self.listed_val_data,
                                              shuffle=False,
                                              validation_labels=self.listed_val_labels,
                                              training_phase=False, context_length=self.context_length)

    def train_dataloader(self):

        return DataLoader(self.train_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=self.shuffle, 
                        num_workers=self.num_workers,
                        pin_memory=True,
                        drop_last=True , 
                        collate_fn=self.dl_collate_fn)

    def val_dataloader(self):

        return DataLoader(self.val_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        num_workers=self.num_workers, 
                        pin_memory=True,
                        drop_last=False, 
                        collate_fn=self.dl_collate_fn)

    def dl_collate_fn(self, batch):

        wsi_names = [row[0] for row in batch]
        image_tensor = torch.stack([row[1] for row in batch])
        text_tensor = torch.stack([row[2] for row in batch])
        next_sequence = torch.stack([row[3] for row in batch])

        return wsi_names, image_tensor, text_tensor, next_sequence
