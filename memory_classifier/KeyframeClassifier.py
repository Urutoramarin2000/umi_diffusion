import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision import models

class KeyframeClassifier_small_lowdim(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # self.vit = ViTModel.from_pretrained("../data/outputs/ckpt/pretrained/vit-base-patch16-224-in21k")
        self.classifier = nn.Sequential(
            nn.Linear(d_model+1, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出 True / False
        )
        for param in self.vit.parameters():
            param.requires_grad = False  
        # for param in self.vit.encoder.layer[-8:].parameters(): 
        #     param.requires_grad = True

    def forward(self, img, pose_velocity):
        vit_outputs = self.vit(pixel_values=img).last_hidden_state[:, 0, :]  
        features = torch.cat([vit_outputs, pose_velocity], dim=1)
        logits = self.classifier(features)
        return logits

class KeyframeClassifier_big_lowdim(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_fc = nn.Linear(d_model, 128)

        self.width_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  
        )
        for param in self.vit.parameters():
            param.requires_grad = False  

    def forward(self, img, lowd_dim):
        """
        img: (B, 3, 224, 224)  # 图像
        pose_velocity: (B, 12)  # 位姿+速度
        """
        vit_outputs = self.vit(pixel_values=img).last_hidden_state[:, 0, :]  
        img_feature = self.vit_fc(vit_outputs)
        # lowd_dim = lowd_dim.unsqueeze(1)
        width_feature = self.width_fc(lowd_dim)
        features = torch.cat([img_feature, width_feature], dim=1)
        # flattened_features = features.view(features.size(0), -1) 

        # transformer_out = self.transformer(features).mean(dim=1)  # Transformer 计算融合
        # logits = self.classifier(vit_outputs)
        logits = self.classifier(features)
        return logits
    

class KeyframeClassifier_no_lowdim(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.vit = ViTModel.from_pretrained("../data/outputs/ckpt/pretrained/vit-base-patch16-224-in21k")
        self.img_feat_dim = 768  

        self.classifier = nn.Sequential(
            nn.Linear(d_model , 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  
        )
        for param in self.vit.parameters():
            param.requires_grad = False  

    def forward(self, img, pose_velosity):
        vit_outputs = self.vit(pixel_values=img).last_hidden_state[:, 0, :]  
        logits = self.classifier(vit_outputs)
        return logits

class KeyframeClassifier_resnet(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.img_fc = nn.Linear(512, 128)

        self.width_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  
        )

        for param in self.resnet_backbone.parameters():
            param.requires_grad = False
        for param in list(self.resnet_backbone.parameters())[-10:]:
            param.requires_grad = True
    def forward(self, img, lowd_dim):

        x = self.resnet_backbone(img)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # 展平成 (B, 512)
        img_feature = self.img_fc(x)  # (B, 128)

        # lowd_dim = lowd_dim.unsqueeze(1)  # (B, 1, 1)
        width_feature = self.width_fc(lowd_dim)  # (B, 32)

        features = torch.cat([img_feature, width_feature], dim=1)  # (B, 160)
        logits = self.classifier(features)
        return logits