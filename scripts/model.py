from torch import nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, nb_classes, n_channels, img_size=(255, 255)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool2d((2, 2), stride=2)
        vector_size = img_size[0] // 2 * img_size[1] // 2 * 64
        self.fc = nn.Linear(vector_size, nb_classes)
        self.nb_classes = nb_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(len(x), -1)
        return self.fc(x)