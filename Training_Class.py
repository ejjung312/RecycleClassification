import os
import torch
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

from Model_Class_From_the_Scratch import MODEL_From_Scratch
from Model_Class_Transfer_Learninig_MobileNet import MobileNet
from Dataset_Class import PyTorch_Classification_Dataset_Class as Dataset

class PyTorch_Classification_Training_Class():
    def __init__(self,
                 dataset_dir="Recycle_Classification_Dataset",
                 batch_size=16,
                 train_ratio=0.75):
        # 데이터세트가 경로에 없는 경우 다운로드
        if not os.path.isdir(dataset_dir):
            os.system("git clone https://github.com/JinFree/Recycle_Classification_Dataset.git")
            os.system("rm -rf ./Recycle_Classification_Dataset/.git")
            dataset_dir = os.path.join(os.getcwd(), "Recycle_Classification_Dataset")
        
        self.USE_CUDA = torch.cuda.is_available()
        # CUDA를 사용할 수 있는 경우 GPU를 이용햔 학습을 수행
        self.DEVICE = torch.device("cuda" if self.USE_CUDA else "cpu")

        # 전처리
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])

        # Dataset 클래스 생성
        dataset = Dataset(dataset_dir=dataset_dir, transform=self.transform)
        # 라벨 파일 저장
        dataset.__save_label_map__()
        # 데이터세트를 훈련용/검증용 분류
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 훈련/검증용 데이터로더 생성
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 신경망 모델 변수 선언. 모델을 저장할 파일명 변수 선언
        self.model = None
        self.model_str = None

    def prepare_network(self, is_scratch=True):
        if is_scratch:
            # is_scratch=True -> 직접 학습
            self.model = MODEL_From_Scratch(self.num_classes)
            self.model_str = "PyTorch_Training_From_Scratch"
        else:
            # is_scratch=False -> MobileNetv2로 학습
            self.model = MobileNet(self.num_classes)
            self.model_str = "PyTorch_Transfer_Learning_MobileNet"

        # GPU를 사용할 수 있는 경우 모델은 GPU로 연산
        self.model.to(self.DEVICE)
        self.model_str += ".pt"

    def training_network(self, learning_rate=0.0001, epochs=10, step_size=3, gamma=0.3):
        if self.model is None:
            self.prepare_network(False)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # step_size만큼의 에포크가 지날 때마다 학습률에 gamma를 곱하는 학습률 스케줄러
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        acc = 0.0

        for epoch in range(1, epochs+1):
            # 훈련
            self.model.train()

            for data, target in tqdm(self.train_loader):
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)

                # 학습을 위한 미분값을 0으로 초기화
                optimizer.zero_grad()

                # 모델 계산 수행
                output = self.model(data)

                # 손실함수 계산
                loss = F.cross_entropy(output, target)

                # 역전파
                loss.backward()

                # 최적화
                optimizer.step()
        
            # 학습률 스케줄러 업데이트
            scheduler.step()

            # 검증
            self.model.eval()

            test_loss = 0
            correct = 0

            with torch.no_grad():
                for data, target in tqdm(self.test_loader):
                    data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                    output = self.model(data)

                    # 교차 엔트로피를 통해 손실값 계산(누적)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()

                    # 예측값 계산
                    pred = output.max(1, keepdim=True)[1]

                    # correct 값 계산
                    correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(self.test_loader.dataset)
            test_accuracy = 100. * correct / len(self.test_loader.dataset)
            print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))

            # 모델 저장
            if acc < test_accuracy or epoch == epochs:
                acc = test_accuracy
                torch.save(self.model.state_dict(), self.model_str)
                print("model saved!")

# print(__name__)
if __name__ == "__main__":
    training_class = PyTorch_Classification_Training_Class()
    training_class.prepare_network(False)
    training_class.training_network()