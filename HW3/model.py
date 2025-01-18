import torch.nn as nn
from utils import Torch_global_set
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
import seaborn as sns
from sklearn.metrics import confusion_matrix

Torch_global_set()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelTmp(nn.Module):
    def __init__(self):
        super(ModelTmp, self).__init__()
        # fixed seed for reproductive results
        torch.manual_seed(114514)
        self.layer1 = nn.Linear(400, 128, bias=True) # construct layer 1
        self.activation1 = nn.ReLU()                 # construct activation 1
        self.layer2 = nn.Linear(128, 64, bias=True)  # construct layer 2
        self.activation2 = nn.ReLU()                 # construct activation 2
        self.layer3 = nn.Linear(64, 4, bias=False)   # construct layer 3
        self.activation3 = nn.Softmax(dim=-1)        # construct activation 3

    def forward(self, x):
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.activation3(self.layer3(x))
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        torch.manual_seed(42)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class OneLayerNN(nn.Module):
    def __init__(self):
        super(OneLayerNN, self).__init__()
        self.f = nn.Linear(28*28, 10)
        # randomly initialize the weights
        nn.init.normal_(self.f.weight, mean=0, std=1)

    def forward(self, x):
        x = self.f(x)
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # fix the random seed for reproductive results
        torch.manual_seed(42)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(val_loader), correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(test_loader), correct / total

def train_test(model, num_epochs, train_loader, val_loader, optimizer, criterion, save_path, device):
    train_loss = []
    val_loss = []
    val_accuracy = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        ts = train(model, train_loader, optimizer, criterion, device)
        vl, va = validate(model, val_loader, criterion, device)

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), save_path)

        train_loss.append(ts)
        val_loss.append(vl)
        val_accuracy.append(va)

        print(f"Epoch [{epoch+1}/{num_epochs}]: Train loss: {ts:.4f}, Val loss: {vl:.4f}, Val accuracy: {va:.4f}")

    plt.plot(range(1, num_epochs+1), train_loss, label='Train')
    plt.plot(range(1, num_epochs+1), val_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def load_image(val_loader):
    average_image = {i: torch.zeros((28,  28)) for i in range(10)}
    l_count = {i: 0 for i in range(10)}

    with torch.no_grad():
        for images, labels in val_loader:
            for i in range(10):
                # mask classifier
                mask = (labels == i)
                average_image[i] += torch.sum(images[mask], dim=0).view(28, 28)
                l_count[i] += mask.sum()
            
    for j in range(10):
        average_image[j] /= l_count[j]

    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(average_image[i].numpy(), cmap='gray')
        plt.title(f'Average Image-Label {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_image_before_training(model_one_layer):
    print('Before training:')
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    weights = model_one_layer.weight.data
    weights = weights.numpy()
    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(weights[i*5+j].reshape(28, 28), cmap='gray')
            axs[i, j].set_title(f'Label {i*5+j+1}')
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()

def feed_forward(train_loader, val_loader):
    model_one_layer = nn.Linear(28*28, 10)
    criterion_one_layer = nn.BCEWithLogitsLoss()
    optimizer_one_layer = optim.SGD(model_one_layer.parameters(), lr=0.01)
    
    model_one_layer.to(device)
    epoch = 0
    while True:
        model_one_layer.train()
        for images, labels in train_loader:
            optimizer_one_layer.zero_grad()
            images = images.view(-1, 28*28).to(device)  # 移动图像到 GPU
            outputs = model_one_layer(images)
            label_one_hot = torch.zeros_like(outputs).to(device)  # 创建并移动 one-hot 编码标签到 GPU
            label_one_hot.scatter_(1, labels.unsqueeze(1).to(device), 1)  # 确保 labels 也在 GPU 上
            loss = criterion_one_layer(outputs, label_one_hot)
            loss.backward()
            optimizer_one_layer.step()

        epoch += 1

        model_one_layer.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(-1, 28*28).to(device)  # 移动图像到 GPU
                outputs = model_one_layer(images)
                label_one_hot = torch.zeros_like(outputs).to(device)  # 创建并移动 one-hot 编码标签到 GPU
                label_one_hot.scatter_(1, labels.unsqueeze(1).to(device), 1)  # 确保 labels 也在 GPU 上
                val_loss += criterion_one_layer(outputs, label_one_hot).item()

        val_loss /= len(val_loader)
        if val_loss < 0.1:
            break

    for i in range(10):
        # 从 GPU 移动到 CPU 并转换为 NumPy
        w = model_one_layer.weight.data[i].view(28, 28).cpu().numpy()
        plt.subplot(2, 5, i+1)
        plt.imshow(w, cmap='gray')
        plt.title(f'Label {i}')
        plt.axis('off')
    plt.show()

def run_cnn(model, train_loader, val_loader, test_loader, optimizer, criterion, path):
    model_simple_cnn = SimpleCNN().to(device)
    optimizer_simple_cnn = optim.SGD(model_simple_cnn.parameters(), lr=0.01)
    criterion_simple_cnn = nn.CrossEntropyLoss()
    # Summary of the model
    print('(1): First have a look at the number of parameters in my CNN:')
    summary(model_simple_cnn, input_size=(1, 28, 28))

    # Train and test processes assuming they handle the device appropriately
    print('(2) & (3) results are shown as following:')
    num_epochs = 10
    path_cnn = 'best_cnn.pt'
    train_test(model, num_epochs, train_loader, val_loader, optimizer, criterion, path, device)

    # Load the model for testing
    model_simple_cnn_ = torch.load(path_cnn, map_location=device)
    model_simple_cnn_.eval()

    # Testing the model
    testl_cnn, testa_cnn = test(model_simple_cnn_, test_loader, criterion_simple_cnn, device)

    print('(4): Results on test set')
    print("Loss on test set:", round(testl_cnn, 4))
    print("Rate of correct prediction on test set:", f'{100 * testa_cnn:.2f}%')

    # Calculating the confusion matrix
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            output_cnn = model_simple_cnn_(images)
            _, pre = torch.max(output_cnn, 1)
            all_labels.extend(labels.cpu())
            all_predictions.extend(pre.cpu())

    all_labels = torch.tensor(all_labels)
    all_predictions = torch.tensor(all_predictions)

    print('(5): Confusion matrix on test set:')
    conf_m = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(conf_m, annot=True, fmt="d", cmap="Reds", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()