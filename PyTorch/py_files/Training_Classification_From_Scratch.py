import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from Model_Scratch import MODEL as MODEL_SCRATCH
from Dataset_Classification import PyTorch_Classification_Dataset

def main(_root_dir = "/content/cats_and_dogs_filtered/train"
         , _epochs = 50, _batch_size = 16):
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    img_width, img_height = 224, 224
    EPOCHS     = _epochs
    BATCH_SIZE = _batch_size
    transform_train = transforms.Compose([
                transforms.Resize(size=(img_width, img_height))
                , transforms.RandomRotation(degrees=15)
                , transforms.ToTensor()
                ])
    transform_test = transforms.Compose([
                transforms.Resize(size=(img_width, img_height))
                , transforms.ToTensor()
                ])

    TrainDataset = PyTorch_Classification_Dataset
    TestDataset = PyTorch_Classification_Dataset

    train_data = TrainDataset(root_dir = _root_dir
                    , transform = transform_train)
    test_data = TestDataset(root_dir = _root_dir
                    , transform = transform_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_data
        , batch_size=BATCH_SIZE
        , shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data
        , batch_size=BATCH_SIZE
        , shuffle=True
    )
    
    train_data.__save_label_map__()
    num_classes = train_data.__num_classes__()

    model = MODEL_SCRATCH(num_classes).to(DEVICE)
    model_str = "PyTorch_Classification_Model_Scratch"
    model_str += ".pt" 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, target in (train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in (test_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)

                # 배치 오차를 합산
                test_loss += F.cross_entropy(output, target,
                                            reduction='sum').item()

                # 가장 높은 값을 가진 인덱스가 바로 예측값
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)

        print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                epoch, test_loss, test_accuracy))

        if acc < test_accuracy:
            acc = test_accuracy
            torch.save(model.state_dict(), model_str)
            print("model saved!")

if __name__ == "__main__":
    main("/content/cats_and_dogs_filtered/train", 50, 16)