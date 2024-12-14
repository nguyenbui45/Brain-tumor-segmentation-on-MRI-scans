from torch.nn import Module,Conv2d,Linear,MaxPool2d,ReLU,ConvTranspose2d,Dropout2d
from torchvision.transforms import v2,CenterCrop
from torch import cat

class Unet(Module):
    def __init__(self,
                num_class=3):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.head = Conv2d(in_channels=64,out_channels=num_class,kernel_size=1,padding='same')
        #self.output_size = output_size
        
    def forward(self,x):
        encoder_feature = self.encoder(x)
        decoder_feature = self.decoder(encoder_feature[0],encoder_feature[1:])
        
        map = self.head(decoder_feature)
        
        return map
        
        
        
class Encoder(Module):
    def __init__(self,channels=(64,128,256,512,1024),input_channel=3):
        super().__init__()
        #block 1 64
        self.conv1_1 = Conv2d(in_channels=input_channel,out_channels=channels[0],kernel_size=3,padding='same')
        self.conv1_2 = Conv2d(in_channels=channels[0],out_channels=channels[0],kernel_size=3,padding='same')
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.dropout1 = Dropout2d(p=0.5)
        
        #block 2 128
        self.conv2_1 = Conv2d(in_channels=channels[0],out_channels=channels[1],kernel_size=3,padding='same')
        self.conv2_2 = Conv2d(in_channels=channels[1],out_channels=channels[1],kernel_size=3,padding='same')
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.dropout2 = Dropout2d(p=0.5)
        
        # block 3 256
        self.conv3_1 = Conv2d(in_channels=channels[1],out_channels=channels[2],kernel_size=3,padding='same')
        self.conv3_2 = Conv2d(in_channels=channels[2],out_channels=channels[2],kernel_size=3,padding='same')
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.dropout3 = Dropout2d(p=0.5)
        
        #block 4 512
        self.conv4_1 = Conv2d(in_channels=channels[2],out_channels=channels[3],kernel_size=3,padding='same')
        self.conv4_2 = Conv2d(in_channels=channels[3],out_channels=channels[3],kernel_size=3,padding='same')
        self.maxpool4 = MaxPool2d(kernel_size=2)
        self.dropout4 = Dropout2d(p=0.5)
        
        #block 5 1024
        self.conv5_1 = Conv2d(in_channels=channels[3],out_channels=channels[4],kernel_size=3,padding='same')
        self.conv5_2 = Conv2d(in_channels=channels[4],out_channels=channels[4],kernel_size=3,padding='same')
        self.maxpool5 = MaxPool2d(kernel_size=2)
        self.dropout5 = Dropout2d(p=0.5)
        
        self.reLU = ReLU()
        
    def forward(self,x):
        x = self.conv1_1(x)
        x = self.reLU(x)
        x = self.conv1_2(x)
        x_crop1 = self.reLU(x)
        x = self.dropout1(x)
        x = self.maxpool1(x_crop1)
        
        x = self.conv2_1(x)
        x = self.reLU(x)
        x = self.conv2_2(x)
        x_crop2 = self.reLU(x)
        x = self.dropout2(x)
        x = self.maxpool2(x_crop2)
        
        x = self.conv3_1(x)
        x = self.reLU(x)
        x = self.conv3_2(x)
        x_crop3 = self.reLU(x)
        x = self.dropout3(x)
        x = self.maxpool3(x_crop3)
        
        x = self.conv4_1(x)
        x = self.reLU(x)
        x = self.conv4_2(x)
        x_crop4 = self.reLU(x)
        x = self.dropout4(x)
        x = self.maxpool4(x_crop4)
        
        x = self.conv5_1(x)
        x = self.reLU(x)
        x = self.conv5_2(x)
        x = self.dropout5(x)
        x = self.reLU(x)
        
        return [x,x_crop4,x_crop3,x_crop2,x_crop1]
    
    
class Decoder(Module):
    def __init__(self,channels=(1024,512,256,128,64)):
        super().__init__()
        #block 512
        self.deconv1 = ConvTranspose2d(in_channels=channels[0],out_channels=channels[1],kernel_size=2,padding=0,stride=2)
        self.conv1_1 = Conv2d(in_channels=channels[0],out_channels=channels[1],kernel_size=3,padding='same')
        self.conv1_2 = Conv2d(in_channels=channels[1],out_channels=channels[1],kernel_size=3,padding='same')
        self.dropout6 = Dropout2d(p=0.5)
        
        #block 256
        self.deconv2 = ConvTranspose2d(in_channels=channels[1],out_channels=channels[2],kernel_size=2,padding=0,stride=2)
        self.conv2_1 = Conv2d(in_channels=channels[1],out_channels=channels[2],kernel_size=3,padding='same')
        self.conv2_2 = Conv2d(in_channels=channels[2],out_channels=channels[2],kernel_size=3,padding='same')
        self.dropout7 = Dropout2d(p=0.5)
        
        
        #block 128
        self.deconv3 = ConvTranspose2d(in_channels=channels[2],out_channels=channels[3],kernel_size=2,padding=0,stride=2)
        self.conv3_1 = Conv2d(in_channels=channels[2],out_channels=channels[3],kernel_size=3,padding='same')
        self.conv3_2 = Conv2d(in_channels=channels[3],out_channels=channels[3],kernel_size=3,padding='same')
        self.dropout8 = Dropout2d(p=0.5)
        
        
        #block 64
        self.deconv4 = ConvTranspose2d(in_channels=channels[3],out_channels=channels[4],kernel_size=2,padding=0,stride=2)
        self.conv4_1 = Conv2d(in_channels=channels[3],out_channels=channels[4],kernel_size=3,padding='same')
        self.conv4_2 = Conv2d(in_channels=channels[4],out_channels=channels[4],kernel_size=3,padding='same')
        self.dropout9 = Dropout2d(p=0.5)
         
        self.reLU = ReLU()
       
    def crop(self,x,copy_feature):
        (_,_,height,width) = x.shape
        cropped_feature = CenterCrop([height,width])(copy_feature)
        return cropped_feature
        
    def forward(self,x,cropped_feature):
       x = self.deconv1(x)
       x = self.reLU(x)
       #crop1
       x_crop = self.crop(x,cropped_feature[0])
       #concat1
       x = cat([x_crop,x],dim=1)
       x = self.conv1_1(x)
       x = self.reLU(x)
       x = self.conv1_2(x)
       x = self.reLU(x)
       x = self.dropout6(x)
       
       x = self.deconv2(x)
       x = self.reLU(x)
       #crop2
       x_crop = self.crop(x,cropped_feature[1])
       #concat2
       x = cat([x_crop,x],dim=1)
       x = self.conv2_1(x)
       x = self.reLU(x)
       x = self.conv2_2(x)
       x = self.reLU(x)
       x = self.dropout7(x)
       
       x = self.deconv3(x)
       x = self.reLU(x)
       #crop3
       x_crop = self.crop(x,cropped_feature[2])
       #concat3
       x = cat([x_crop,x],dim=1)
       x = self.conv3_1(x)
       x = self.reLU(x)
       x = self.conv3_2(x)
       x = self.reLU(x)
       x = self.dropout8(x)
       
       x = self.deconv4(x)
       x = self.reLU(x)
       #crop3
       x_crop = self.crop(x,cropped_feature[3])
       #concat4
       x = cat([x_crop,x],dim=1)
       x = self.conv4_1(x)
       x = self.reLU(x)
       x = self.conv4_2(x)
       x = self.reLU(x)
       x = self.dropout9(x)
       
       return x