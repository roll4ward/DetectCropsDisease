from utils import train_test_split, os, shutil
def split_dataset(data_dir, test_size=0.2, valid_size=0.25):
  class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

  for class_dir in class_dirs:
      class_path = os.path.join(data_dir, class_dir)
      files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

      train_files, temp_files = train_test_split(files, test_size=test_size + valid_size, random_state=42)
      valid_files, test_files = train_test_split(temp_files, test_size=test_size / (test_size + valid_size), random_state=42)

      train_dir = os.path.join(data_dir, 'train',class_dir)
      valid_dir = os.path.join(data_dir, 'valid',class_dir)
      test_dir = os.path.join(data_dir, 'test',class_dir)

      os.makedirs(train_dir, exist_ok=True)
      os.makedirs(valid_dir, exist_ok=True)
      os.makedirs(test_dir, exist_ok=True)

      # 파일을 각각의 디렉토리로 이동합니다.
      for f in train_files:
          shutil.move(f, os.path.join(train_dir, os.path.basename(f)))
      for f in valid_files:
          shutil.move(f, os.path.join(valid_dir, os.path.basename(f)))
      for f in test_files:
          shutil.move(f, os.path.join(test_dir, os.path.basename(f)))

      if not os.listdir(class_path):
        os.rmdir(class_path)
# 예제 사용법
if __name__ == '__main__':
    data_dir = 'C:\\Users\\mirun\\PycharmProjects\\detect_classify\\datasets\\mangofruitdds\\SenMangoFruitDDS_original'
    split_dataset(data_dir)
