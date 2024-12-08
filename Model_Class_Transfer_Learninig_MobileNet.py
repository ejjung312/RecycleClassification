import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        # 전이학습을 위해 MobileNet v2 모델 사용
        # 이미지넷으로 학습한 가중치를 불러옴(pretrained=True)
        self.network = models.mobilenet_v2(pretrained=pretrained)
        
        # MobileNet의 마지막 레이어가 num_classes만큼의 클래스를 분류할 수 있게 수정
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(num_ftrs, num_classes)
        self.classifier = nn.Sequential(nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.network(x)
        x = self.classifier(x)
        return x