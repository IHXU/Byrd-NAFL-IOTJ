import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    标准的 LeNet-5 模型实现。
    
    默认配置适配 MNIST/Fashion-MNIST (1通道, 28x28 输入)。
    为了保持 LeNet-5 原生结构的特征图大小，我们在 Forward 中加入了 padding。
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(LeNet5, self).__init__()
        
        # C1: 卷积层 (Input -> 6 feature maps)
        # 输入: (B, 1, 32, 32) (经过 padding 后)
        # 输出: (B, 6, 28, 28)
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        
        # S2: 池化层 (Subsampling)
        # 输入: (B, 6, 28, 28)
        # 输出: (B, 6, 14, 14)
        self.pool = nn.MaxPool2d(2, 2)
        
        # C3: 卷积层
        # 输入: (B, 6, 14, 14)
        # 输出: (B, 16, 10, 10)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # S4: 池化层 (复用 self.pool)
        # 输入: (B, 16, 10, 10)
        # 输出: (B, 16, 5, 5)
        
        # C5: 全连接层 (作为卷积层或线性层实现均可，这里使用线性层)
        # 输入: 16 * 5 * 5 = 400
        # 输出: 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        # F6: 全连接层
        # 输入: 120
        # 输出: 84
        self.fc2 = nn.Linear(120, 84)
        
        # Output: 输出层
        # 输入: 84
        # 输出: num_classes (10)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 适配输入尺寸
        # MNIST 标准是 28x28，LeNet-5 原文期望 32x32。
        # 这里进行 padding 填充 (左, 右, 上, 下) 各填充 2 像素
        if x.shape[-1] == 28:
            x = F.pad(x, (2, 2, 2, 2))
            
        # C1 -> ReLU -> S2
        x = self.pool(F.relu(self.conv1(x)))
        
        # C3 -> ReLU -> S4
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten 展平: (B, 16, 5, 5) -> (B, 400)
        x = x.view(-1, 16 * 5 * 5)
        
        # C5 (FC) -> ReLU
        x = F.relu(self.fc1(x))
        
        # F6 -> ReLU
        x = F.relu(self.fc2(x))
        
        # Output (Logits)
        # 注意：不在此处加 Softmax，因为 CrossEntropyLoss 会包含 Softmax
        x = self.fc3(x)
        
        return x

def lenet5(num_classes=10):
    """便捷构造函数"""
    return LeNet5(num_classes=num_classes)
