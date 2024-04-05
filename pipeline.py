
from ultralytics import YOLO
import os
from torchvision import transforms
from PIL import Image
import torch


def input_image(image_path, custom_model_path):
    model = YOLO('yolov8n-seg.pt')  # load an official model
    model = YOLO(custom_model_path)  # load a custom model
    img = Image.open(image_path)
    model(img)


def extract_to_tensor():
    directory="../segmentation/runs/segment"
    folders = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    folders.sort(key=os.path.getctime, reverse=True)  # 가장 최근에 생성된 순으로 정렬
    latest_folder = folders[0]
    crops_folder=os.path.join(latest_folder,"crops")

    all_data = []
    for type in os.listdir(crops_folder):
        for file in os.listdir(os.path.join(crops_folder, type)):
            img_path = os.path.join(crops_folder, type, file)  # 이미지 파일 경로 수정
            img = Image.open(img_path)
            transform = transforms.ToTensor()
            img_tensor = transform(img)
            label = type
            all_data.append((img_tensor, label))
    torch.save(all_data, 'all_image_labels.pt')
    return all_data



