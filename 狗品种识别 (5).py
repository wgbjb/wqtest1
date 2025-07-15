import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights
# 训练模型
class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedClassifier, self).__init__()
        # 加载基础模型
        self.base_model = models.efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        
        # 获取特征维度
        num_features = self.base_model.classifier[1].in_features
        
        # 修改注意力机制的命名
        self.attention = nn.Sequential()
        self.attention.add_module('fc', nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        ))
        
        # 分类器保持不变
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # 特征提取
        features = self.base_model.features(x)
        features = self.base_model.avgpool(features)
        features = torch.flatten(features, 1)
        
        # 注意力机制
        attention_weights = self.attention.fc(features)
        features = features * attention_weights
        
        # 分类
        return self.classifier(features)
        
    def load_state_dict(self, state_dict, strict=False):
        """处理加载不同结构的状态字典"""
        model_state_dict = self.state_dict()
        new_state_dict = {}
        
        # 只加载匹配的层
        for k, v in state_dict.items():
            # 检查键是否在当前模型中且形状匹配
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"跳过加载权重 {k}: 形状不匹配或键不存在")
                
        # 调用父类的load_state_dict
        return super().load_state_dict(new_state_dict, strict=False)
    
    @staticmethod
    def init_weights(m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)