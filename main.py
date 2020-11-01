import multiprocessing

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from transforms import ConstantPad
from dataloader import FridgeVoterDataset
from model import CnnModel

num_workers = multiprocessing.cpu_count()
epochs = 5

if __name__ == '__main__':
    transform = ConstantPad(shape=(700,600,3),padding_mode='reflect')
    dataset = FridgeVoterDataset('dataset', 'data.json', transform=transform)
    train_len = int(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    trainloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=num_workers)

    print(train_set[0][0].shape)
    model = CnnModel(train_set[0][0].shape)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            image, vote = data
            labels = FridgeVoterDataset.vote_to_class(vote)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')