import multiprocessing

from torch.utils.data import DataLoader

from dataloader import FridgeVoterDataset

num_workers = multiprocessing.cpu_count()

if __name__ == '__main__':
    dataset = FridgeVoterDataset('dataset', 'data.json')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=num_workers)