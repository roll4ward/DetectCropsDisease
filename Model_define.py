from utils import *
class CustomResNet50(nn.Module):
  def __init__(self, pretrained=True, num_classes=10, freeze_layers=True):
    super(CustomResNet50, self).__init__()
    self.model=models.resnet50(pretrained=pretrained)
    num_ftrs = self.model.fc.in_features
    self.model.fc=nn.Linear(num_ftrs, num_classes)

    if freeze_layers:
      self.freeze_layers()

  def freeze_layers(self):
      ct = 0
      for child in self.model.children():
          ct += 1
          if ct < 6:
              for param in child.parameters():
                  param.requires_grad = False

  def forward(self, x):
      return self.model(x)
