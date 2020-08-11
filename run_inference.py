import torch
from model import Generator
import numpy as np
import torchvision.utils as utils
import os
import matplotlib.pyplot as plt
import pickle


def run_inference(model_dir=None, model_path=None):
    hp_file = model_dir + 'hp.pkl'
    f = open(hp_file, "rb")
    hp = pickle.load(f)

    os.makedirs(hp['save_image_dir'], exist_ok=True)
    generator = Generator()
    ckpt = torch.load(model_dir+model_path)
    generator.load_state_dict(ckpt['G'])

    with torch.no_grad():
        z = torch.randn(1, 100, 1, 1)
        generated_image = generator(z)

        plt.axis("off")
        plt.title("Generated Image")
        plt.imshow(np.transpose(utils.make_grid(generated_image, padding=2, normalize=True), (1, 2, 0)))
        plt.savefig(hp['save_image_dir'] + '/image.jpeg')
        plt.show()


if __name__ == '__main__':
    model_dir = ""  # add model directory eg: checkpoints/DCGAN_20200811_123456/"
    model_path = ""  # add model path eg: dc_gan_weights_epoch50.pth
    run_inference(model_dir, model_path)