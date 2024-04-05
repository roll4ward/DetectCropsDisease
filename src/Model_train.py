from utils import *
from Model_define import CustomResNet50
class Model_trainer:
  def __init__(self, num_classes=10, freeze_layers=True,dataset_sizes=None):
    self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.num_classes=num_classes
    self.freeze_layers = freeze_layers
    self.dataset_sizes = dataset_sizes
    self.model, self.criterion, self.optimizer, self.scheduler = self.initialize_model()


  def initialize_model(self):
      model = CustomResNet50(pretrained=True, num_classes=self.num_classes, freeze_layers=self.freeze_layers)
      model = model.to(self.device)

      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
      exp_lr_scheduler = lr_scheduler.StepLR(optimizer , step_size=7, gamma=0.1)

      return model, criterion, optimizer , exp_lr_scheduler


  def train_resnet(self,loader, num_epochs=10):
    best_model_wts=copy.deepcopy(self.model.state_dict())
    best_acc=0

    for epoch in range(num_epochs):
      print(f"----------epoch {epoch+1}-----------")
      since=time.time()

      for phase in["train", "valid"]:
        if phase=="train":
          self.model.train()
        else:
          self.model.eval()
        running_loss=0.0
        running_corrects=0.0

        for inputs, labels in loader[phase]:
          inputs = inputs.to(self.device)
          labels=labels.to(self.device)

          self.optimizer.zero_grad()

          with torch.set_grad_enabled(phase=="train"):
            outputs=self.model(inputs)
            x, preds=torch.max(outputs,1)
            loss=self.criterion(outputs, labels)

            if phase=="train":
              loss.backward()
              self.optimizer.step()
          running_loss+=loss.item()*inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)
        if phase =="train":
          self.scheduler.step()
          l_r=[x["lr"] for x in self.optimizer .param_groups]
          print("learning_rate : ",l_r)

        epoch_loss=running_loss/self.dataset_sizes[phase]
        epoch_acc=running_corrects.double() /self.dataset_sizes[phase]
        print("{} Loss: {:4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

        if phase =="valid" and epoch_acc>best_acc:
          best_acc=epoch_acc
          best_model_wts=copy.deepcopy(self.model.state_dict())
      time_elapsed=time.time()-since
      print("Completed in {:.0f}m {:0f}s".format(time_elapsed // 60, time_elapsed % 60))

    print("Best valid Acc: {:.4f}".format(best_acc))

    self.model.load_state_dict(best_model_wts)

    return self.model
