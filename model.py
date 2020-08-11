import torch.nn as nn
import torch


class Generator(nn.Module):

    def __init__(self, z=100):
        super(Generator, self).__init__()
        self.z = z
        self.gen_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z, out_channels=1024, kernel_size=(4, 4), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, z):
        generated_image = self.gen_layers(z)
        return generated_image


class Discriminator (nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.dis_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.Sigmoid()
        )

    def forward(self, image):
        prediction = self.dis_layers(image)
        prediction = torch.reshape(prediction, [-1, 1])
        return prediction


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, prediction, target):
        loss = self.bce_loss(prediction, target)
        return loss