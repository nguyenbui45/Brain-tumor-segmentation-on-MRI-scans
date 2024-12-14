import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
from torchsummary import summary
import segmentation_models_pytorch as smp
from metrics import binary_dice_loss,binary_dice_coeff
import logging
import numpy as np
import time
import os


from Unet import Unet 
from PreprocessingData import Load_Brain_MRI_dataset, read_file_path

class segmentation_BrainMRI:
    def __init__(self,saved_model_path,image_path,mask_path,image_shape):
        self.HEIGHT = image_shape[0]
        self.WIDTH = image_shape[1]
        self.CHANNEL = image_shape[2]
        self.image_path  = image_path
        self.mask_path = mask_path
        self.saved_model_path = saved_model_path
        return
    
    
    def train(self,loss,optimizer,device,num_epochs,batch_size,saved_name,log_name):
        #define logging file
        logging.basicConfig(
            filename=log_name,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        
        transform = v2.Compose([
            v2.ToTensor(),
            v2.Resize((self.HEIGHT,self.WIDTH)),
            v2.ToDtype(torch.float32,scale=True)   
        ])
        
        mask_transform = v2.Compose([
            v2.ToTensor(),
            v2.Resize((self.HEIGHT,self.WIDTH), antialias=False, interpolation=InterpolationMode.NEAREST),
            v2.ToDtype(torch.float32,scale=True)   
        ])
        
        train_img_list,train_mask_list = read_file_path(self.image_path,'train/') 
        validation_img_list,validation_mask_list = read_file_path(self.mask_path,'validation/') 
        test_img_list = read_file_path(self.image_path,'test/')
        
        self.train_dataset = DataLoader(Load_Brain_MRI_dataset(train_img_list[:],
                                                               train_mask_list[:],
                                                               transform,
                                                               mask_transform),
                                        batch_size=batch_size,
                                        shuffle=True)
        
        self.validation_dataset = DataLoader(Load_Brain_MRI_dataset(validation_img_list,
                                                               validation_mask_list,
                                                               transform,
                                                               mask_transform),
                                             batch_size=batch_size,
                                             shuffle=True)
        
        sample,mask = next(iter(self.train_dataset))
        #print(f'Shape of 1 batch sample:{sample[1,:,:,:].unique()}')
        #print(f'Shape of 1 batch mask:{mask[1,:,:,:].unique()}')
        
        # choose the hardware to train mode
        self.device = torch.device(device)
        
        # Load Unet model
        #self.model = Unet(num_class=1)
        self.model = smp.Unet('efficientnet-b6',encoder_weights='imagenet',in_channels=3,classes=1)
        logging.info(self.model) #record the model strucure to log file
        self.model = self.model.to(self.device)
        #summary(self.model,(3,128,128))
        
        match loss:
            case "crossentropy":
                self.criterion = nn.CrossEntropyLoss()
            case "BCE":
                self.criterion = nn.BCEWithLogitsLoss()
                
        
        match optimizer:
            case "SGD":
                self.optimizer = optim.SGD(self.model.parameters(),lr=0.001,momentum=0.9)
            case "Adam":
                self.optimizer = optim.Adam(self.model.parameters(),lr=0.001,weight_decay=1e-8)
                
                
        self.num_epochs = int(num_epochs)
        
        # take milestone of trainng time
        since = time.time()
        best_model_params_path = os.path.join(self.saved_model_path,saved_name)
        best_loss=100.0
        best_score=0.0
        
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs-1}')
            print('-'*10)
            logging.info(f'Epoch {epoch}/{self.num_epochs-1}')           
            logging.info('-'*10)
            
            # train phase
            self.model.train()
            
            running_loss = 0.0
            running_dice_score=0.0
            
            for images,masks in self.train_dataset:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                #forward pass
                with torch.set_grad_enabled(True):
                    outputs = self.model(images)
                    loss = self.criterion(outputs,masks)
                    #loss +=binary_dice_loss(outputs,masks)
                    dice_score = binary_dice_coeff(outputs,masks,threshold=0.5)
                    
                    loss.backward() # calculate the gradient w.r.t loss
                    self.optimizer.step() # back prop
                    
                running_loss += loss.item()* images.size(0)
                running_dice_score += dice_score.item() * images.size(0)
                
        

            epoch_loss = running_loss / len(train_img_list)
            epoch_dice_score = running_dice_score / len(train_img_list)
            print(f'Training loss: {epoch_loss:.4f}')
            print(f'Training dice score: {epoch_dice_score:.4f}')
            
            
            # validation phase
            self.model.eval()
            running_loss = 0.0
            running_dice_score = 0.0
            
            for images,masks in self.validation_dataset:
                images = images.to(device)
                masks = masks.to(device)
                
                with torch.set_grad_enabled(False):
                    outputs = self.model(images)
                    loss = self.criterion(outputs,masks)
                    #loss+=binary_dice_loss(outputs,masks)
                    dice_score = binary_dice_coeff(outputs,masks,threshold=0.5)
                    
                running_loss+=loss.item() * images.size(0)
                running_dice_score+=dice_score.item()*images.size(0)
                
            
            epoch_loss = running_loss / len(validation_img_list)
            epoch_dice_score = running_dice_score / len(validation_img_list)
            print(f'Validation loss: {epoch_loss:.4f}')
            print(f'Validation dice score: {epoch_dice_score:.4f}')
            logging.info(f'Validation loss: {epoch_loss:.4f}')
            logging.info(f'Validation dice score: {epoch_dice_score:.4f}')
            
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_score = epoch_dice_score
                torch.save(self.model.state_dict(),best_model_params_path)
                
            print()
            
            
        time_elapsed = time.time() - since
        
        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best validation loss: {best_loss:4f}')
        print(f'Best validation score: {best_score:4f}')
    
        logging.info(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
        logging.info(f'Best validation loss: {best_loss:4f}')
        logging.info(f'Best validation score: {best_score:4f}')
        return
            
        
         
         
         
        

