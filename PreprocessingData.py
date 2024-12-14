import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from segmentation_models_pytorch import metrics
import os
from utilities import sorted_alphanumeric
from PIL import Image

class Load_Brain_MRI_dataset(Dataset):
    def __init__(self,image_path,mask_path,transform,mask_transform):
        self.image_path = image_path
        self.mask_path = mask_path
        self.dataset = {"image":[],"mask":[]}
        self.transform = transform
        self.mask_transform = mask_transform
        
        return self.load_dataset(image_path,mask_path)
    
    
    def load_dataset(self,image_path,mask_path):
        '''
        Description: load image and mask to memory
        Input: 
            + image_path: (array of str) list of images destination
            + mask_path: (array of str) list of mask destination
            
        Output: None
        
        '''
        for row in range(0,len(image_path)):
            image = Image.open(image_path[row])
            #print(np.array(image).shape)
            image_tensor = torch.from_numpy(np.array(image))
            image_tensor = image_tensor.permute(2,0,1)
            
            mask = Image.open(mask_path[row])
            #print(np.array(mask).shape)
            mask = np.expand_dims(mask,axis=-1)  # expand the image from (H,W) shape to (H,W,1) shape to conform transform method
            mask_tensor = torch.from_numpy(np.array(mask))
            mask_tensor = mask_tensor.permute(2,0,1)
            
        
            self.dataset['image'].append(image_tensor)
            self.dataset['mask'].append(mask_tensor)    
    
    
    
    def __getitem__(self,idx):
        image = self.dataset['image'][idx]
        mask = self.dataset['mask'][idx]
        
        image = self.transform(image)
        mask = self.mask_transform(mask)
        #print(mask.unique())
        return image,mask
    
    def __len__(self):
        return len(self.dataset['mask'])
    
    
def read_file_path(path,folder_name):
    '''
    Description: read the path of image and mask of dataset
    Input: 
        + path: (str) the path to the dataset
        + folder_name: (str) the name of folder to read. There are 3 options: 'train/', 'test/', 'test/'
        
    Output:
        + img_path,mask_path: (list) string array that stores the path of each image-mask pair
    '''
    
    MASK_PREFIX = '_mask'
    img_path = []
    mask_path = []
    if folder_name != 'test/':
        for folder in (os.listdir(path + folder_name)):
            current_path = path + folder_name + folder + '/'
            for sub_folder in (sorted_alphanumeric(os.listdir(current_path))):
                if MASK_PREFIX in sub_folder:
                    mask_path.append(current_path + sub_folder) 
                elif MASK_PREFIX not in sub_folder:
                    img_path.append(current_path + sub_folder)
                    
        return img_path,mask_path 
    else:
        for folder in (os.listdir(path + folder_name)):
            current_path = path + folder_name + folder + '/'
            for sub_folder in (sorted_alphanumeric(os.listdir(current_path))):
                img_path.append(current_path + sub_folder)
        
        return img_path
        
        