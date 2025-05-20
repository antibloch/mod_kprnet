import torch
import torch.nn as nn
import cv2
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.sequence = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.sequence(x)
    
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels)
        )
    def forward(self, x):
        return self.sequence(x)
    


class UNet(nn.Module):
    def __init__(self, in_channels_coarse=1,in_channels_fine=1, out_channels=1):
        super(UNet, self).__init__()
        #input_dim = 256
        self.encoder_coarse = nn.ModuleList([
            DownConv(in_channels_coarse, 64), #128
            DownConv(64, 128), #64
            DownConv(128, 256), #32
            DownConv(256, 512) #16
        ])
        self.encoder_fine = nn.ModuleList([
            DownConv(in_channels_fine, 64), #128
            DownConv(64, 128), #64
            DownConv(128, 256), #32
            DownConv(256, 512) #16
        ])
        self.bottleneck = ConvBlock(512+512, 1024)
        #extra channels allow for concatenation of skip connections in upsampling block
        self.decoder = nn.ModuleList([
            UpConv(512+1024,512), #32
            UpConv(256+512,256), #64
            UpConv(128+256,128), #128
            UpConv(64+128,64) #256
        ])
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    def forward(self, x1,x2):
        o1 = x1
        for layer in self.encoder_coarse:
            o1 = layer(o1)
        skips = []
        o2 = x2
        for layer in self.encoder_fine:
            o2 = layer(o2)
            skips.append(o2)
        o = torch.cat((o1,o2), dim=1)
        o = self.bottleneck(o)
        for i, layer in enumerate(self.decoder):
            #print(o.size())
            o = torch.cat((skips[len(skips)-i-1],o), dim=1)
            #print(o.size())
            o = layer(o)
        return self.output_conv(o)
def coarse_filter_image(image):
    image=0.2989*image[0]+0.5870*image[1]+0.1140*image[2]
    # image=torch.unsqueeze(image, 0)
    # convert from torch to numpy
    img = image.numpy()
    size1=2
    gauss_kernel = np.ones((size1,size1),np.float32)/(size1*size1)
    imager = cv2.filter2D(img,-1,gauss_kernel)
    size2=3
    fil_kernel = np.ones((size2,size2), np.uint8)
    img_dilation = cv2.dilate(imager, fil_kernel, iterations=2)
    img_erosion = cv2.erode(img_dilation, fil_kernel, iterations=5+2)
    image = cv2.GaussianBlur(img_erosion, (7, 7), 0)
    # Apply Sobel filter for edge detection
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Sobel Edge Detection on the Y axis
    # Absolute values and combine the edges
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    # thresholding img such that if sobel_combined[i,j] > 0.7*max(sobel_combined) then sobel_combined[i,j] = 255 else 0
    max_sobel = np.max(sobel_combined)
    img_copy = img.copy()
    for i in range(sobel_combined.shape[0]):
        for j in range(sobel_combined.shape[1]):
            if sobel_combined[i,j] > 0.1*max_sobel:
                img_copy[i,j] = img_copy[i,j]
            else:
                img_copy[i,j] = 0.3*img_copy[i,j]
    img_copy= torch.from_numpy(img_copy).float()
    img_copy=torch.unsqueeze(img_copy, 0)
    return img_copy


def fine_filter_image(image):
    image=0.2989*image[0]+0.5870*image[1]+0.1140*image[2]
    image = image.numpy()
    image = cv2.GaussianBlur(image, (7, 7), 0)
    # Apply Sobel filter for edge detection
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Sobel Edge Detection on the Y axis
    # Absolute values and combine the edges
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    sobel_combined = torch.from_numpy(sobel_combined).float()
    sobel_combined=torch.unsqueeze(sobel_combined, 0)
    return sobel_combined





if __name__ == "__main__":
    input_1 = torch.randn(1,1,1024,256)
    input_2 = torch.randn(1,1,1024,256)
    device = torch.device('cuda')
    model = UNet(out_channels=10).to(device)
    input_1 = input_1.to(device)
    input_2 = input_2.to(device)
    print(f"Input 1 shape: {input_1.shape}")
    print(f"Input 2 shape: {input_2.shape}")
    print(f"Output shape: {model(input_1, input_2).shape}")