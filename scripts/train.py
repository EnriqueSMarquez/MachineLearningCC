import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from model import SimpleCNN
from dataset import CustomDataset
from tqdm import tqdm
import json

train_images_path = '../CIFAR-10/train/'
train_df_path = '../CIFAR-10/real_train.csv'
val_images_path = '../CIFAR-10/train/'
val_df_path = '../CIFAR-10/real_val.csv'
saving_path = './run1.json'

batch_size = 32
labels_to_index = json.load(open('../CIFAR-10/labels2index.json'))

training_dataset = CustomDataset(train_images_path, train_df_path,
                                 labels_to_index, transforms=transforms.ToTensor())
validation_dataset = CustomDataset(val_images_path, val_df_path,
                                   labels_to_index, transforms=transforms.ToTensor())

train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(len(labels_to_index), 3, (32, 32))
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print('Starting train')
metrics = {'train_loss' : [], 'train_acc' : [], 'val_loss' : [], 'val_acc' : []}
for epoch_index in range(10):
    running_loss, running_corrects = 0.0, 0
    model.train()
    print('Starting epoch')
    for batch_x, batch_y in tqdm(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        predictions = torch.max(outputs, 1)[1]
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
        running_corrects += torch.sum(predictions == batch_y.data)
    metrics['train_loss'] += [running_loss / len(training_dataset)]
    metrics['train_acc'] += [(running_corrects.double() / len(training_dataset)).item()]
    print(f'Training loss {metrics["train_loss"][-1]}')
    print(f'Training Accuracy {metrics["train_acc"][-1]}')

    running_loss, running_corrects = 0.0, 0
    print('Starting validation')
    model.eval()

    for batch_x, batch_y in tqdm(val_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, batch_y)
        running_loss += loss.item() * batch_x.size(0)
        running_corrects += torch.sum(preds == batch_y.data)
    metrics['val_loss'] += [running_loss / len(validation_dataset)]
    metrics['val_acc'] += [(running_corrects.double() / len(validation_dataset)).item()]
    print(f'Validation loss {metrics["val_loss"][-1]}')
    print(f'Validation Accuracy {metrics["val_acc"][-1]}')

with open(saving_path, 'w') as f:
    json.dump(metrics, f)