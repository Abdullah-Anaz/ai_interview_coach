import torch
import torch.nn as nn 
import numpy as np
import math
import copy
import torchvision.models as models

def resnet18(pretrained=False, maps=32):
    """
    ResNet-18 backbone matching the exact state_dict keys and channel dimensions 
    of the official GazeTR checkpoint.
    """
    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    
    class GazeTRBackbone(nn.Module):
        def __init__(self, resnet_net, target_maps):
            super().__init__()
            # Map directly to standard torchvision names present in the checkpoint
            self.conv1 = resnet_net.conv1
            self.bn1 = resnet_net.bn1
            self.relu = resnet_net.relu
            self.maxpool = resnet_net.maxpool
            self.layer1 = resnet_net.layer1
            self.layer2 = resnet_net.layer2
            self.layer3 = resnet_net.layer3
            self.layer4 = resnet_net.layer4
            
            # Matches checkpoint's base_model.conv structure (512 -> target_maps channels)
            self.conv = nn.Sequential(
                nn.Conv2d(512, target_maps, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(target_maps)
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.conv(x)
            return x

    return GazeTRBackbone(base, maps)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7*7
        dim_feedforward=512
        dropout = 0.1
        num_layers=6

        self.base_model = resnet18(pretrained=False, maps=maps)

        # d_model: dim of Q, K, V 
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(
                  maps, 
                  nhead, 
                  dim_feedforward, 
                  dropout)

        encoder_norm = nn.LayerNorm(maps) 
        # num_encoder_layer: deeps of layers 

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        self.feed = nn.Linear(maps, 2)
            
        self.loss_op = nn.L1Loss()


    def forward(self, x_in):
        feature = self.base_model(x_in["face"])
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)
        
        position = torch.from_numpy(np.arange(0, 50)).cuda()

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)
  
        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0]

        gaze = self.feed(feature)
        
        return gaze


    def extract_latents(self, x_in):
        """
        Bypasses the final regression layer to return the 32-dimensional 
        Transformer cls_token embedding for downstream late fusion.
        """
        feature = self.base_model(x_in["face"])
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.repeat((1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)
        
        # Dynamically map position tensor to the current device instead of hardcoding .cuda()
        position = torch.from_numpy(np.arange(0, 50)).to(feature.device)
        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)
  
        feature = feature.permute(1, 2, 0)
        
        # Extract the cls_token representation [batch_size, maps(32)]
        latent_embedding = feature[:, :, 0]

        return latent_embedding


    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label) 
        return loss
