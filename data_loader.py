from utils import *

class GetMeanStd():
    def __init__(self, dataset_path, batch_size=64):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

    def calculate(self):
        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _ in self.dataloader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        return mean.numpy().tolist(), std.numpy().tolist()


from torchvision import datasets, transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root, mean, std, train=True):
        if train:
            self.transform = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(52),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            # 테스트 데이터일 경우 적용할 변환
            self.transform = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.CenterCrop(52),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        self.dataset = datasets.ImageFolder(root=root, transform=self.transform)
        self.classes = self.dataset.classes

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

    def __len__(self):
        return len(self.dataset)


