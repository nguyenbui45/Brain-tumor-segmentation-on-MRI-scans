from Brain_MRI_segmentation import segmentation_BrainMRI
import argparse
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--optimizer",default="Adam",type=str)
    parser.add_argument("-l","--loss",default="BCE",type=str)
    parser.add_argument("-d","--device",default="cuda:0",type=str)
    parser.add_argument("-e","--epochs",default=30,type=int)
    parser.add_argument('-b','--batchsize',default=64,type=int)
    parser.add_argument('-sn','--savedname',default="Unet_ENet_Adam.pt",type=str)
    parser.add_argument('-lf','--logfile',default="Unet_ENet_Adam.log",type=str)
    args = parser.parse_args()
    
    image_path = "/home/nguyensolbadguy/Code_Directory/DL_Implementation/computer_vision/Brain_MRI/"

    label_path = "/home/nguyensolbadguy/Code_Directory/DL_Implementation/computer_vision/Brain_MRI/"
    
    saved_model_path =  "/home/nguyensolbadguy/Code_Directory/DL_Implementation/computer_vision/Brain_MRI/trained_model/Unet/"
    
    main = segmentation_BrainMRI(saved_model_path=saved_model_path,image_path=image_path,mask_path=label_path,image_shape=[128,128,3])
    
    
    main.train(args.loss,args.optimizer,args.device,args.epochs,args.batchsize,args.savedname,args.logfile)