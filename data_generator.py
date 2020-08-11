from torch.utils.data.dataset import Dataset
import numpy as np
import os
from torchvision import transforms
from PIL import Image


class DataGenerator(Dataset):
    def __init__(self, hp):
        super(DataGenerator, self).__init__()
        self.hp = hp
        self.total_train_images = hp['total_train_images']
        self.train_data_path = hp['train_data_path']
        self.train_image_file_names = os.listdir(self.train_data_path)

    def __getitem__(self, item):
        np.random.seed()

        # read a random image from the data set
        select_random_img_idx = np.random.randint(len(self.train_image_file_names))
        image = Image.open(self.train_data_path+self.train_image_file_names[select_random_img_idx])

        # crop center image  64 x 64, normalise image betwwen -1 and 1
        transform = transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        image = transform(image)

        return image

    def __len__(self):
        return self.total_train_images
