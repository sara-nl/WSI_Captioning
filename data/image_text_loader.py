from pathlib import Path
from random import randint, choice
import json

import pandas as pd
import os
import torch
import numpy as np
import h5py

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class TextImageDataset(Dataset):
    def __init__(self,
                 wsi_embeddings: dict,
                 text_embeddings: dict,
                 text_tokens: dict,
                 shuffle: bool = True,
                 training_phase: bool = True,
                 context_length: int = 32
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
        """
        super().__init__()
        self.wsi_embeddings = wsi_embeddings
        self.text_embeddings = text_embeddings
        self.text_tokens = text_tokens
        self.shuffle = shuffle
        self.training_phase = training_phase

        self.wsi_names = list(self.wsi_embeddings)

        self.context_length = context_length

        print("Number of Images: ", len(self.wsi_embeddings))        

    def __len__(self):
        return len(self.wsi_embeddings)

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

        key = self.wsi_names[ind]

        visual_embeddings = torch.tensor(self.wsi_embeddings[key])

        # For now, we are taking 32 patches
        chosen_ints = torch.randint(0,visual_embeddings.shape[0], (32,))

        visual_embeddings = visual_embeddings[chosen_ints]
        # the first visual embedding serves as an embeddor for the whole sequence of visual embeddings
        # This is similar to having a CLS token in BERT models
        visual_embeddings[0,:] = torch.ones(visual_embeddings.shape[1])
        
        text_embeddings = torch.tensor(self.text_embeddings[key])
        next_sequence = torch.tensor(self.text_tokens[key])

        token_sequence = next_sequence[1:self.context_length+1]
        embedded_sequence = text_embeddings[:self.context_length]

        return key, visual_embeddings, embedded_sequence, token_sequence

class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 wsi_embeddings_path: str,
                 text_embeddings_path: str,
                 text_tokens_path: str,
                 batch_size: int,
                 dataset: str,
                 num_workers: int = 0,
                 shuffle: bool = True,
                 context_length: int = 32
                 ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataset = dataset
        self.context_length = context_length

        self.train_wsi_embeddings = self.read_hdf5_dataset(wsi_embeddings_path+"train_wsi_embeddings.hdf5")
        self.train_text_embeddings = self.read_hdf5_dataset(text_embeddings_path+"train_text_embeddings.hdf5")
        self.train_text_tokens = self.read_hdf5_dataset(text_tokens_path+"train_target_tokens.hdf5")

        self.val_wsi_embeddings = self.read_hdf5_dataset(wsi_embeddings_path+"val_wsi_embeddings.hdf5")
        self.val_text_embeddings = self.read_hdf5_dataset(text_embeddings_path+"val_text_embeddings.hdf5")
        self.val_text_tokens = self.read_hdf5_dataset(text_tokens_path+"val_target_tokens.hdf5")        

        print("train_data length: ", len(self.listed_train_data))
        print("val_data length: ", len(self.listed_val_data))
        print("validation labels number: ", len(self.listed_val_labels))
    
    def read_hdf5_dataset(self, file_path):
        with h5py.File(file_path, 'r') as f:
            keys = f['dataset/keys'][:]
            values = f['dataset/values'][:]
            
            # Decode keys if they are in binary format
            keys = [key.decode() if isinstance(key, bytes) else key for key in keys]
            
            # Create a dictionary to store the keys-values pairs
            data_dict = {}
            for i in range(len(keys)):
                data_dict[keys[i]] = values[i]
                
        return data_dict

    def setup(self, stage: str):

        print("num workers: ", self.num_workers)
        self.train_dataset = TextImageDataset(self.train_wsi_embeddings,
                                              self.train_text_embeddings,
                                              self.train_text_tokens,
                                              shuffle=self.shuffle,
                                              training_phase=True, context_length=self.context_length)

        self.val_dataset = TextImageDataset(self.train_wsi_embeddings,
                                              self.train_text_embeddings,
                                              self.train_text_tokens,
                                              shuffle=False,
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
