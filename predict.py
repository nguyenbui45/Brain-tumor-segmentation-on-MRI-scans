from PreprocessingData import Load_Brain_MRI_dataset,read_file_path
import argparse
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
from Unet import Unet
import segmentation_models_pytorch as smp
from map_color import map_color
from metrics import binary_dice_loss,binary_dice_coeff



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",default="Unet")
    parser.add_argument("-n","--numimage",default=6)
    parser.add_argument("-d","--device",default="cuda:0") 
    args = parser.parse_args() 
    #load model
    
    PATH = "/home/nguyensolbadguy/Code_Directory/DL_Implementation/computer_vision/Brain_MRI/trained_model/Unet/"
    IMAGE_PATH  = "/home/nguyensolbadguy/Code_Directory/DL_Implementation/computer_vision/Brain_MRI/"
    LABEL_PATH = "/home/nguyensolbadguy/Code_Directory/DL_Implementation/computer_vision/Brain_MRI/"
    
    
    match(args.model):
        case "Unet":
            #model = Unet(num_class=3)
            model = smp.Unet('efficientnet-b6',encoder_weights="imagenet",in_channels=3,classes=1)
            check_point = torch.load(PATH + "Unet_ENet_Adam.pt",weights_only=False)
    
    model.load_state_dict(check_point)
    model.eval() 
    device=torch.device(args.device)
    model.to(device)
   
   # load test set 
    transform = v2.Compose([
                            v2.ToTensor(),
                            v2.Resize((128,128), antialias=False, interpolation=InterpolationMode.NEAREST),
                            v2.ToDtype(torch.float32,scale=True),
        ])
        
    label_transform = v2.Compose([
                            v2.ToTensor(),
                            v2.Resize((128,128), antialias=False, interpolation=InterpolationMode.NEAREST),
                            v2.ToDtype(torch.float32,scale=True),
        ])
    
    
    test_image_list,test_mask_list = read_file_path(IMAGE_PATH,'validation/')
                                 
    test_dataset = DataLoader(Load_Brain_MRI_dataset( test_image_list,
                                                    test_mask_list,
                                                    transform,
                                                    label_transform),
                                                    batch_size=64,
                                                    shuffle=False)

    num_image=0 
    with torch.no_grad():
        for images,labels in test_dataset:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images).squeeze()
            dice_score = binary_dice_coeff(outputs,labels)
            probs = F.sigmoid(outputs)
            prediction = (probs >= 0.5).float()
            prediction = prediction.cpu().numpy()
            
            
            gt = labels.permute(0,2,3,1).squeeze()
            gt = gt.cpu().numpy()
            
            for j in range(images.size()[0]):
                fig = plt.figure()
                
                num_image+=1
                axes = plt.subplot(1,3,num_image)
                axes.axis('off')
                #draw original image
                image = images[j,:,:,:].detach().cpu().numpy()
                image_3channel = (image.transpose(1,2,0)*255).astype(np.uint8)
                axes.imshow(image_3channel)
                axes.set_title("original image")
                num_image +=1
                
                #draw predictions
                pred_mask =prediction[j,:,:]
                pred_mask_3channel = map_color(pred_mask)
                axes = plt.subplot(1,3,num_image)
                axes.axis('off')
                axes.imshow(image_3channel)
                axes.imshow(pred_mask_3channel,alpha=0.3)
                axes.set_title("prediction")
                num_image +=1
                
                #draw ground truth
                gt_mask = gt[j,:,:]
                gt_mask_3channel = map_color(gt_mask)
                axes = plt.subplot(1,3,num_image)
                axes.axis('off')
                axes.imshow(image_3channel)
                axes.imshow(gt_mask_3channel,alpha=0.3)
                axes.set_title("ground truth")
                
                num_image = 0
                
                plt.show()
                #concat 3 image
                #im_show = np.hstack((image_3channel,255*np.ones((image_3channel.shape[0],20,3)),pred_mask_3channel,255*np.ones((image_3channel.shape[0],20,3)),#gt_mask_3channel)).astype(np.uint8)
                                    
                #saved_image = Image.fromarray(im_show)
                #saved_image.save(f"predict_Enet{j}.png")
                