import io
import json
import os
import requests

import skimage.io as io
from torch.utils.data.dataset import Dataset
import torch

def get_data(root_dir, data_json):
    with open(os.path.join(root_dir, data_json)) as f:
        data = json.load(f)

        for qi in data['quiz_items']:
            image_name = qi['file']
            if not os.path.exists(os.path.join(root_dir, image_name)):
                print(f'Getting image {image_name}')
                url = f'https://static01.nyt.com/newsgraphics/2020/08/24/fridge-politics/assets/images/fridges/big/{image_name}'
                r = requests.get(url)

                with open(os.path.join(root_dir, image_name), 'wb') as img:
                    img.write(r.content)

        return data['quiz_items']

class FridgeVoterDataset(Dataset):

    def __init__(self, root_dir, data_json):
        self.data_list = get_data(root_dir, data_json)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.data_list[idx]
        img_name = entry['file']
        answer = entry['answer']

        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)

        return {'image': image, 'vote' : answer}

if __name__ == '__main__':
    dataset = FridgeVoterDataset('dataset', 'data.json')
    dataset[0]