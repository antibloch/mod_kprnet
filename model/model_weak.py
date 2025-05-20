import torch
import torch.nn as nn
import cv2
import numpy as np




class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()
        
        self.conv_1 = self.conv_block(input_channels, 32)
        self.conv_2 = self.conv_block(32, 64)
        self.conv_3 = self.conv_block(64, 128)
        self.conv_4 = self.conv_block(128, 256)
        self.conv_5 = self.conv_block(256, 512)
        
        self.upsample_6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_6 = self.conv_block(512, 256)
        
        self.upsample_7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_7 = self.conv_block(256, 128)
        
        self.upsample_8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_8 = self.conv_block(128, 64)
        
        self.upsample_9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_9 = self.conv_block(64, 32)
        
        self.output_conv = nn.Conv2d(32, output_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        
        conv_1 = self.conv_1(x)
        maxp_1 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_1)
        
        conv_2 = self.conv_2(maxp_1)
        maxp_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_2)
        
        conv_3 = self.conv_3(maxp_2)
        maxp_3 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_3)
        
        conv_4 = self.conv_4(maxp_3)
        maxp_4 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_4)
        
        conv_5 = self.conv_5(maxp_4)
        upsample_6 = self.upsample_6(conv_5)
        
        concat_6 = torch.cat((upsample_6, conv_4), dim=1)
        conv_6 = self.conv_6(concat_6)
        upsample_7 = self.upsample_7(conv_6)
        
        concat_7 = torch.cat((upsample_7, conv_3), dim=1)
        conv_7 = self.conv_7(concat_7)
        upsample_8 = self.upsample_8(conv_7)
        
        concat_8 = torch.cat((upsample_8, conv_2), dim=1)
        conv_8 = self.conv_8(concat_8)
        upsample_9 = self.upsample_9(conv_8)
        
        concat_9 = torch.cat((upsample_9, conv_1), dim=1)
        conv_9 = self.conv_9(concat_9)
        
        outputs = self.output_conv(conv_9)
        return outputs.squeeze(1)  # Remove the added channel dimension