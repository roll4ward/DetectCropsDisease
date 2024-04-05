from data_loader import GetMeanStd, CustomDataset
from Model_define import CustomResNet50
from Model_train import Model_trainer
from utils import *
if __name__ == '__main__':
    data_dir = 'C:\\Users\\mirun\\PycharmProjects\\detect_classify\\datasets\\mangofruitdds\\SenMangoFruitDDS_original'
    get_mean_std = GetMeanStd(os.path.join(data_dir,'train'))
    mean, std = get_mean_std.calculate()

    # CustomDataset 인스턴스 생성
    custom_dataset = {x : CustomDataset(root=os.path.join(data_dir, x), mean=mean, std=std) for x in ["train", "valid"]}

    # DataLoader 인스턴스 생성
    dataloader = {x: DataLoader(custom_dataset[x], batch_size=64, shuffle=True, num_workers=4) for x in ["train", "valid"]}
    dataset_sizes = {x: len(custom_dataset[x]) for x in ["train", "valid"]}
    class_names = custom_dataset["train"].classes


    model=CustomResNet50()
    # ModelTrainer 인스턴스 생성 및 학습

    trainer = Model_trainer(num_classes=len(class_names), freeze_layers=True, dataset_sizes=dataset_sizes)

    model_resnet50 = trainer.train_resnet(dataloader, num_epochs=20)
    torch.save(model_resnet50, "../datasets/mangofruitdds/SenMangoFruitDDS_original/fruit_disease.pt")