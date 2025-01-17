"""
The architecture of SiamFC
Written by Heng Fan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import *


class SiamNet(nn.Module):

    def __init__(self):
        super(SiamNet, self).__init__()

        # architecture (AlexNet like)
        self.feat_extraction = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),             # conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),  # conv2, group convolution
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),           # conv3
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2), # conv4
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)  # conv5
        )

        # adjust layer as in the original SiamFC in matconvnet
        self.adjust = nn.Conv2d(1, 1, 1, 1)

        # initialize weights
        self._initialize_weight()

        self.config = Config()

    def forward(self, z, x):
        """
        forward pass
        z: examplare, BxCxHxW
        x: search region, BxCxH1xW1
        """
        # get features for z and x
        z_feat = self.feat_extraction(z)
        x_feat = self.feat_extraction(x)

        # correlation of z and z
        xcorr_out = self.xcorr(z_feat, x_feat)

        score = self.adjust(xcorr_out)

        return score
    
    def get_normalised_score(self, z, x):
        z_feat = self.feat_extraction(z)
        x_feat = self.feat_extraction(x)

        bz, chz, hz, wz = z_feat.size()
        bx, chx, hx, wx = x_feat.size()

        for i in range(bz):
            z_feat[i] /= torch.norm(z_feat[i])
        xcorr_out = self.xcorr(z_feat, x_feat)

        prevh, prevw = None, None
        for i in range(bx):

            for h in range(0, hx-hz+1):
                sum_of_squares = torch.sum(x_feat[i, :, h: h+hz, 0: 0+wz] ** 2)

                for w in range(0, wx-wz+1):
                    if w:
                        addw = torch.sum(x_feat[i, :, h: h+hz, w+wz-1: w+wz] ** 2)
                        sum_of_squares += addw - prevw

                    xcorr_out[i, :, h, w] /= torch.sqrt(sum_of_squares)
                    prevw = torch.sum(x_feat[i, :, h: h+hz, w: w+1] ** 2)

        return xcorr_out

    def xcorr(self, z, x):
        """
        correlation layer as in the original SiamFC (convolution process in fact)
        """
        batch_size_x, channel_x, w_x, h_x = x.shape
        x = torch.reshape(x, (1, batch_size_x * channel_x, w_x, h_x))

        # group convolution
        out = F.conv2d(x, z, groups = batch_size_x)

        batch_size_out, channel_out, w_out, h_out = out.shape
        xcorr_out = torch.reshape(out, (channel_out, batch_size_out, w_out, h_out))

        return xcorr_out

    def _initialize_weight(self):
        """
        initialize network parameters
        """
        tmp_layer_idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tmp_layer_idx = tmp_layer_idx + 1
                if tmp_layer_idx < 6:
                    # kaiming initialization
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                else:
                    # initialization for adjust layer as in the original paper
                    m.weight.data.fill_(1e-3)
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_loss(self, prediction, label, weight):
        """
        weighted cross entropy loss
        """
        return F.binary_cross_entropy_with_logits(prediction,
                                                  label,
                                                  weight,
                                                  size_average=False) / self.config.batch_size
    
    def focal_loss(self, prediction, label, weight, gamma = 2.): 
        prediction = prediction * (2 * label - 1) # if label[idx]=0 then prediction[idx]=-prediction[idx] otherwise remains same
        pt = prediction.sigmoid()
        modulating_factor = (1. - pt).pow(gamma)
        loss = weight * modulating_factor * pt.log()
        return -1*loss.sum()/label.size()[0]
