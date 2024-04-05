from data_loader import GetMeanStd, CustomDataset
from Model_define import CustomResNet50
from Model_train import Model_trainer
from utils import *
import torch
import torch.nn.functional as F
def evaluate(model, test_loader):
  model.eval() # 모델 평가모드로 변경
  test_loss = 0
  correct = 0

  with torch.no_grad():
    for data, target in test_loader:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      data, target = data.to(device), target.to(device)
      output = model(data)

      test_loss += F.cross_entropy(output, target, reduction = "sum").item()  # loss값은 교차엔트로피 총합으로! -> default는 mean
      pred = output.max(1, keepdim = True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
  test_loss /= len(test_loader.dataset) # 이전 test_loss는 배치별 loss총합이 더해진 값. 이를 배치 개수로 나누어 평균 계산
  test_accuracy = 100. * correct / len(test_loader.dataset)
  return test_loss, test_accuracy

if __name__ == '__main__':
    data_dir = 'C:\\Users\\mirun\\PycharmProjects\\detect_classify\\datasets\\mangofruitdds\\SenMangoFruitDDS_original'

    get_mean_std = GetMeanStd(os.path.join(data_dir, 'train'))
    mean, std = get_mean_std.calculate()

    # 테스트 데이터셋 생성
    test_dataset = CustomDataset(root=os.path.join(data_dir, 'test'), mean=mean, std=std, train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
    class_names = test_dataset.classes

    # 모델 로드
    resnet_model = torch.load(data_dir + "/resnet50.pt")
    resnet_model.eval()


    # 모델 평가
    test_loss, test_accuracy = evaluate(resnet_model, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")



