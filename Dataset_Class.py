import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

# class PyTorch_Custom_Dataset_Class(Dataset):
#     def __init__(self):
#         super().__init__()
#         pass
#     def __getitem__(self, idx):
#         pass
#     def __len__(self):
#         pass

class PyTorch_Classification_Dataset_Class(Dataset):
    def __init__(self
                 , dataset_dir = "Recycle_Classification_Dataset"
                 , transform = None):
        super().__init__()

        # 데이터세트가 지정한 경로에 있는지 확인
        if not os.path.isdir(dataset_dir):
            # 데이터세트가 지정한 경로에 없다면 다운로드
            os.system("git clone https://github.com/JinFree/Recycle_Classification_Dataset.git")
            os.system("rm -rf ./Recycle_Classification_Dataset/.git")

        self.image_abs_path = dataset_dir
        self.transform = transform

        if self.transform is None:
            # 이미지 전처리
            self.transform = transforms.Compose([
                transforms.Resize(256), # 256x256 사이즈로 조정
                transforms.RandomCrop(224), # 랜덤하게 224x224 영역 추출
                transforms.ToTensor(), # 0~1 사이의 값을 가진 텐서로 변환
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 정규화
            ])

        # 입력받은 경로 바로 아래에 있는 폴더의 이름이 분류할 클래스의 이름
        self.label_list = os.listdir(self.image_abs_path)
        self.label_list.sort()
        self.x_list = []
        self.y_list = []

        # image_abs_path 경로 내 모든 폴더를 순차적으로 확인
        for label_index, label_str in enumerate(self.label_list):
            # 이미지만 있다고 가정하고 폴더 내 모든 파일의 경로를 리스트로 제작
            img_path = os.path.join(self.image_abs_path, label_str)
            img_list = os.listdir(img_path)

            for img in img_list:
                self.x_list.append(os.path.join(img_path, img))
                self.y_list.append(label_index)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        # PIL 모듈의 Image 함수를 활용해 이미지 오픈
        image = Image.open(self.x_list[idx])

        # 이미지가 RGB가 아닐 경우(흑백이미지) RGB로 변경
        if image.mode is not "RGB":
            image = image.convert("RGB")
        
        # 데이터세트를 신경망에 입력하기 전에 전처리 수행
        if self.transform is not None:
            image = self.transform(image)

        return image, self.y_list[idx]

    def __save_label_map__(self, dst_text_path = "label_map.txt"):
        label_list = self.label_list
        f = open(dst_text_path, "w")
        for i in range(len(label_list)):
            f.write(label_list[i]+'\n')
        f.close()

    def __num_classes__(self):
        return len(self.label_list)