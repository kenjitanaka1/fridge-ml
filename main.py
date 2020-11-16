from datetime import datetime
import multiprocessing
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from transforms import ConstantPad
from dataloader import FridgeVoterDataset
from model import CnnModel

num_workers = multiprocessing.cpu_count()
epochs = 5
save_dir = './saved_models'

if __name__ == '__main__':
    transform = ConstantPad(shape=(3, 700, 600), padding_mode='reflect')
    dataset = FridgeVoterDataset('dataset', 'data.json', transform=transform)
    train_len = int(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    trainloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=num_workers)
    
    model = CnnModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0
    for epoch in range(epochs):
        for i, (image, vote) in enumerate(trainloader, 0):
            # image, vote = data
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
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    dt_string = datetime.now().strftime('%d%m%Y-%H%M%S')
    torch.save(model, os.path.join(save_dir, f'trained-model-{dt_string}.pt'))

    # model = torch.load('saved_models\\trained-model-14112020-113411.pt')
    print('Finished Training')
    model.eval()
    correct = 0
    for _, (image, vote) in enumerate(testloader):
        guess = model(image)
        guess = np.argmax(guess.detach().numpy(), axis=1)
        labels = FridgeVoterDataset.vote_to_class(vote)

        correct += np.sum(guess == labels.numpy())

    print(f'Accuracy: {correct}/{len(test_set)}={correct/len(test_set)*100}%')
    # test 
