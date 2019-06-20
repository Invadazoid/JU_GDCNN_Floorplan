# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import torch

# SEGNET
class SegNet(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)
        x12_s = x12.shape
        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)
        x22_s = x22.shape
        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)
        x32_s = x32.shape
        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)
        x42_s = x42.shape
        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)
        x52_s = x52.shape
        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2,output_size=x52_s)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))
        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2,output_size=x42_s)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2,output_size=x32_s)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2,output_size=x22_s)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2,output_size=x12_s)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        return x11d

# FCN
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

# This is implemented in full accordance with the original one (https://github.com/shelhamer/fcn.berkeleyvision.org)
class FCN8s(nn.Module):
    def __init__(self, num_classes, pretrained=False, caffe=False):
        super(FCN8s, self).__init__()
        vgg = models.vgg16()
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        '''
        100 padding for 2 reasons:
            1) support very small input size
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        '''
        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore_pool4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

    def forward(self, x):
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)
        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)
        score_pool3 = self.score_pool3(0.0001 * pool3)
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                 + upscore_pool4)
        return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()
    
# PSPNET
        
def initialize_weights(model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            if s!=1:
                self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim, momentum=.95),
                    nn.ReLU(inplace=True)
                ))
            if s==1:
                self.features.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    #nn.BatchNorm2d(reduction_dim, momentum=.95),
                    nn.ReLU(inplace=True)
                ))                
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, use_aux=True):
        super(PSPNet, self).__init__()
        self.use_aux = use_aux
        resnet = models.resnet101(pretrained=True)
#        if pretrained:
#            resnet.load_state_dict(torch.load(res101_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if use_aux:
            self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
            #print(self.aux_logits)
            initialize_weights(self.aux_logits)

        initialize_weights(self.ppm)
        initialize_weights(self.final)

    def forward(self, x):
        x_size = x.size()

        x = self.layer0(x)
    
        x = self.layer1(x)
        
        x = self.layer2(x)

        x = self.layer3(x)

        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)

        x = self.ppm(x)

        x = self.final(x)
        if self.training and self.use_aux:
            return F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:], mode='bilinear')
        return F.upsample(x, x_size[2:], mode='bilinear')

# TIRAMISU    
class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)
    
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)
    
def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #print("x ",x.size())
        out = self.firstconv(x)
        #print("out ",out.size())
        skip_connections = []
        #print("going in loop")
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            #print("out (dense blocks down) ",out.size())
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
            #print("out (trans down blocks) ",out.size())
        
        #print("out of loop")
        out = self.bottleneck(out)
        #print("out bottleneck ",out.size())
        
        #print("going in loop")
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            #print("out trans up block ",out.size())
            out = self.denseBlocksUp[i](out)
            #print("out dense block up ",out.size())
        #print("out of loop")
        out = self.finalConv(out)
        #print("out final conv",out.size())
        #out = self.softmax(out)
        #print("out softmax ",out.size())
        return out


def FCDenseNet57(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet67(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet103(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)

# U-NET

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding = kernel_size // 2)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding = kernel_size // 2)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))

        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size= 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding = kernel_size // 2)#1st parameter in_size
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding = kernel_size // 2)
        self.activation = activation

    def center_crop(self, layer, target_size):
        
        batch_size, n_channels, layer_width, layer_height = layer.size()
        x1 = (layer_width - target_size[0]) // 2
        y1 = (layer_height - target_size[1]) // 2
        return layer[:, :, x1:(x1 + target_size[0]), y1:(y1 + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x,output_size=bridge.size())
        crop1 = self.center_crop(bridge, (up.size()[2],up.size()[3]))   # IMPROVEMENT + ALL PADDING 
        out = torch.cat([up, crop1], 1)
        
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))

        return out

class UNet(nn.Module):
    def __init__(self,no_class):#
        super(UNet, self).__init__()
        #self.imsize = imsize
        self.activation = F.relu
       
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(3, 64)#3--1
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.conv_block512_1024 = UNetConvBlock(512, 1024)

        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)

        self.last = nn.Conv2d(64, no_class, 1)#11--2


    def forward(self, x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        
        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        
        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        
        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)
        
        block5 = self.conv_block512_1024(pool4)

        up1 = self.up_block1024_512(block5, block4)
        up2 = self.up_block512_256(up1, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)

        return self.last(up4)  

# DEEPLAB
        
affine_par = True
def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    i = int(i)
    #print('i',i)
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class Bottleneck1(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation_ = 1, downsample=None):
        super(Bottleneck1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding = padding, bias = False, dilation= dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        #print(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        #print(out)
        return out 
class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = []
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
            for m in self.conv2d_list:
                m.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers,NoLabels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],NoLabels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))
        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(dilation_series,padding_series,NoLabels)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

class MS_Deeplab(nn.Module):
    def __init__(self,block,NoLabels):
	    super(MS_Deeplab,self).__init__()
	    self.Scale = ResNet(block,[3, 4, 23, 3],NoLabels)   #changed to fix #4 
    def forward(self,x):
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size = (  int(input_size*0.75)+1,  int(input_size*0.75)+1  ),mode='bilinear')
        self.interp2 = nn.Upsample(size = (  int(input_size*0.5)+1,   int(input_size*0.5)+1   ),mode='bilinear')
        self.interp3 = nn.Upsample(size = (  outS(input_size),   outS(input_size)   ),mode='bilinear')
        out = []
        out1 = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        a1 = self.Scale(x)
        a0 = F.softmax(self.Scale(x),1)
        out.append(a0)	# for original scale
        out1.append(self.Scale(x))
        y = self.Scale(x2)
        a2 = self.interp3(y)
        z1 = F.softmax(y,1)
    
        out.append(z1)#or 0.75x scale
        out1.append(y)
        a3 = self.Scale(x3)
        a4 = F.softmax(self.Scale(x3),1)
        out.append(a4)
         
        
        out1.append(self.Scale(x3))      
            
        x2Out_interp = a2
        x3Out_interp = self.interp3(a3)
        temp1 = torch.max(a1,x2Out_interp)
        a5 = F.softmax(torch.max(temp1,x3Out_interp),1)
        out.append(a5)
        out1.append(torch.max(temp1,x3Out_interp))
        
        return out,out1
    
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.inplanes = inplanes
        
        self.bn1 = nn.BatchNorm2d(inplanes)#me
        
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.bn2 = nn.BatchNorm2d(squeeze_planes)#me
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
#        self.bn3 = nn.BatchNorm2d(squeeze_planes)#me
#        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
#                                   kernel_size=3, padding=1)
#        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.expand3x3 = conv_dw(squeeze_planes, expand3x3_planes,1)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(self.bn1(x)))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(self.bn2(x))),
            self.expand3x3(x)
        ], 1)
    
    
class UpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
            
        def tconv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.ConvTranspose2d(inp, inp, 2, stride, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
            
        super(UpBlock, self).__init__()
        #self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size= 2, stride=2)
        self.up = tconv_dw(in_size, out_size, 2)
        self.conv = conv_dw(in_size, out_size,1)
#        self.conv_extra = nn.Sequential(    
#                nn.Conv2d(in_size, out_size, 1, 1, 0, bias=False),
#                nn.BatchNorm2d(out_size),
#                nn.ReLU(inplace=True),
#            )

    def forward(self, x, bridge):
        #print("x",x.size())
        #print("bridge",bridge.size())
        #if x.size()[2:] != bridge.size()[2:]:
        up = self.up(x)
        up = F.upsample(up,size=bridge.size()[2:],mode='bilinear',align_corners=True)
        #else:
         #    up = self.conv_extra(x)
        #print("up",up.size())
        crop1 = bridge
        out = torch.cat([up, crop1], 1)
        out = self.conv(out)
        return out

class SqueezeNet(nn.Module):

    def __init__(self, num_classes ,version=1.1 ):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        self.mod1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True)  
            )
        
        self.mod2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            )
        
        self.mod3 = nn.Sequential( 
                nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
               )
               
        
        self.mod4 = nn.Sequential( 
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
            )
        
        self.up3 = UpBlock(512,256)
        self.up2 = UpBlock(256,128)
        self.up1 = UpBlock(128,64)
        self.last = nn.Conv2d(64, self.num_classes, 1)


#        final_conv = nn.Conv2d(64, self.num_classes, kernel_size=1)#512
#        self.classifier = nn.Sequential(
#            nn.Dropout(p=0.5),
#            final_conv,
##            nn.ReLU(inplace=True),
#            #nn.AvgPool2d(13, stride=1)
#        )
        
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                if m is final_conv:
#                    init.normal_(m.weight, mean=0.0, std=0.01)
#                else:
#                    init.kaiming_uniform_(m.weight)
#                if m.bias is not None:
#                    init.constant_(m.bias, 0)

    def forward(self, x):
        #print(x.size())
        x1 = self.mod1(x)
        #print(x.size())
        x2 = self.mod2(x1)
        #print(x.size())
        x3 = self.mod3(x2)
        #print(x.size())
        x4 = self.mod4(x3)
        #print(x4.size())
        up3 = self.up3(x4,x3)
        #print(up3.size())
        up2 = self.up2(up3,x2)
        #print(up2.size())
        up1 = self.up1(up2,x1)
        #print(up1.size())
        
        
        #x = self.classifier(x)
        return F.upsample(self.last(up1),size=x.size()[2:],mode='bilinear',align_corners=True)

class UpBlock_mobile(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
            
        def tconv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.ConvTranspose2d(inp, inp, 2, stride, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
            
        super(UpBlock_mobile, self).__init__()
        #self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size= 2, stride=2)
        self.up = tconv_dw(in_size, out_size, 2)
        self.conv = conv_dw(in_size, out_size,1)

    def forward(self, x, bridge):
        #print("x",x.size())
        #print("bridge",bridge.size())
        up = self.up(x)
        #print("up",up.size())
        
        if bridge.size()[2:] != up.size()[2:]:
            crop1 = F.upsample(bridge,size=up.size()[2:],mode='bilinear', align_corners = True)
        else:
            crop1 = bridge
            
        out = torch.cat([up, crop1], 1)
        out = self.conv(out)
        return out
    
class MobileNet(nn.Module):
    def __init__(self,nc):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.num_classes = nc
        self.a = conv_bn(  3,  32, 2)
        self.b = conv_dw( 32,  64, 1)
        self.c = conv_dw( 64, 128, 2)
        self.d = conv_dw(128, 128, 1)
        self.e = conv_dw(128, 256, 2)
        self.f = conv_dw(256, 256, 1)
        self.g = conv_dw(256, 512, 2)
        self.h = conv_dw(512, 512, 1)
        self.i = conv_dw(512, 512, 1)
        self.j = conv_dw(512, 512, 1)
        self.k = conv_dw(512, 512, 1)
        self.l = conv_dw(512, 512, 1)
        self.m = conv_dw(512, 1024, 2)
        self.n = conv_dw(1024, 1024, 1)
        
        self.up3 = UpBlock_mobile(1024,512)
        self.up2 = UpBlock_mobile(512,256)
        self.up1 = UpBlock_mobile(256,128)
        self.up0 = UpBlock_mobile(128,64)
        self.last = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, x):
        xa = self.a(x)
        #print(xa.size())
        xb = self.b(xa)
        #print(xb.size())
        
        xc = self.c(xb)
        #print(xc.size())
        xd = self.d(xc)
        #print(xd.size())
        
        xe = self.e(xd)
        #print(xe.size())
        xf = self.f(xe)
        #print(xf.size())
        
        xg = self.g(xf)
        #print(xg.size())
        xh = self.h(xg)
        #print(xh.size())
        xi = self.i(xh)
        #print(xi.size())
        xj = self.j(xi)
        #print(xj.size())
        xk = self.k(xj)
        #print(xk.size())
        xl = self.l(xk)
        #print(xl.size())
        
        xm = self.m(xl)
        #print(xm.size())
        xn = self.n(xm)
        #print(xn.size())
        
        up3 = self.up3(xn,xl)
        #print(up3.size())
        up2 = self.up2(up3,xf)
        #print(up2.size())
        up1 = self.up1(up2,xd)
        #print(up1.size())
        up0 = self.up0(up1,xb)
        #print(up0.size())
        return F.upsample(self.last(up0),size=x.size()[2:],mode='bilinear', align_corners = True)


def load_model(model_name,noc):
    if model_name == 'deeplab':
        model = MS_Deeplab(Bottleneck1,noc)
    if model_name=='fcn':
        model = FCN8s(noc)
    if model_name=='segnet':
        model = SegNet(3,noc)
    if model_name=='pspnet':
        model = PSPNet(noc)
    if model_name=='unet':
        model = UNet(noc)
    if model_name=='tiramisu':
        model = FCDenseNet103(noc) 
    if model_name == 'squeezenet':
        model =  SqueezeNet(noc)
    if model_name == 'mobilenet':
        model =  MobileNet(noc)
    return model
