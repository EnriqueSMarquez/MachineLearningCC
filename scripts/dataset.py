import os
import torch
import cv2
import pandas as pd

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, labels_df, labels_to_index, transforms=None):
        self.folder_path = folder_path
        self.labels_df = pd.read_csv(labels_df)
        self.labels_to_index = labels_to_index
        self.transforms = transforms #Augmentation and ToTensor
    
    def __getitem__(self, index):
        '''Read index image + label and return it'''
        input_data = self.labels_df.loc[index, :]
        image = cv2.imread(os.path.join(self.folder_path, input_data['image_name']),
        cv2.IMREAD_UNCHANGED) #Read image
        label = self.labels_to_index[input_data['label']] #Change label to number
        if self.transforms:
            image = self.transforms(image) #Run given transformations / augmentations
        
        return image, label
    def __len__(self):
        return len(self.labels_df)