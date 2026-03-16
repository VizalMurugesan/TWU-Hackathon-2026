# written with AI
import torch.nn as nn

class ChessNet(nn.Module):
    """Small CNN that predicts a move from a board state."""
    def __init__(self, n_classes=4096):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(12, 64 , kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.c3 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(256*8*8, n_classes)

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = h.view(h.size(0), -1)          # flatten → (B, 256×8×8)
        return self.classifier(h)
