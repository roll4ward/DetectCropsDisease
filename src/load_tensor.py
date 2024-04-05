import torch

fruit_disease=['겹무늬병', '탄저병','회색 곰팡이', '정상', '과경부 썩음']
leaf_disease=['a', 'b', 'c', 'd', 'e']
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # 모델을 평가 모드로 설정
    return model

def predict(all_tensor, cla_model, label):
    cla_model.eval()

    img_tensor = all_tensor.unsqueeze(0)  # 크기: [1, 채널, 높이, 너비]
    img_tensor = img_tensor.cuda()
    # 모델에 이미지 넣어 예측 수행
    with torch.no_grad():  # 그래디언트 계산 비활성화
        output = cla_model(img_tensor)

    _, predicted = torch.max(output, 1)

    if "leaf" in label:
        result = leaf_disease[predicted.item()]
    else:
        result = fruit_disease[predicted.item()]
    return result
