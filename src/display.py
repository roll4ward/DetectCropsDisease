from pipeline import input_image, extract_to_tensor
from load_tensor import load_model, predict

if __name__ == "__main__":
    image_path = "G:\\내 드라이브\\aihub_tomato_dataset\\test\\images\\V001_tom1_39_028_e_05_20210930_14_02093010_49122255.png"
    seg_model="G:\\내 드라이브\\aihub_tomato_dataset\\model\\only_tomato\\best.pt"
    classify__models={
        "fruit": "../tensors/fruit_disease.pt",
        "leaf": "../tensors/fruit_disease.pt",
    }
    input_image(image_path, seg_model)
    img_tensors=extract_to_tensor()
    for img_tensor, label in img_tensors:
        if "flower" in label:
            print(f"there is a {label}")
        else:
            model_path = classify__models.get("leaf" if "leaf" in label else "fruit")
            model = load_model(model_path)
            predicted_result = predict(img_tensor, model, label)
            print(f"Predicted: {predicted_result}, Label: {label}")



