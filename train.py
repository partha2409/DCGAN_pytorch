import torch
import numpy as np
from torch.utils.data import DataLoader
import model
from data_generator import DataGenerator
from config import hp
import os
import time
import matplotlib.pyplot as plt
import pickle


def train(pre_trained=None):

    # create folder to save models and loss graphs

    reference = hp['net_type'] + str(time.strftime("_%Y%m%d_%H%M%S"))
    checkpoints_folder = hp["output_dir"] + '/checkpoints/' + reference
    os.makedirs(checkpoints_folder, exist_ok=True)

    # save hyper parameter settings
    pickle_file_location = checkpoints_folder + "/hp.pkl"
    pickle_file = open(pickle_file_location, "wb")
    pickle.dump(hp, pickle_file)
    pickle_file.close()

    # create data iterator
    dataset = DataGenerator(hp)
    iterator = DataLoader(dataset=dataset, batch_size=hp['batch_size'], num_workers=hp['num_workers'], pin_memory=True, shuffle=False, drop_last=True)

    # create model and loss

    generator = model.Generator().to(device)
    discriminator = model.Discriminator().to(device)

    loss = model.BCELoss().to(device)

    # optimizer

    g_optimizer = torch.optim.Adam(params=generator.parameters(), lr=hp['learning_rate'], betas=(hp['beta1'], hp['beta2']))
    d_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=hp['learning_rate'], betas=(hp['beta1'], hp['beta2']))

    start_epoch = 0

    # load pre trained model

    if pre_trained is not None:
        ckpt = torch.load(pre_trained)
        generator.load_state_dict(ckpt['G'])
        discriminator.load_state_dict(ckpt['D'])
        g_optimizer.load_state_dict(ckpt['G_opt'])
        d_optimizer.load_state_dict(ckpt['D_opt'])
        start_epoch = ckpt['epoch'] + 1

    # init loss arrays

    generator_loss = np.zeros(hp['num_epochs'])
    discriminator_loss = np.zeros(hp['num_epochs'])

    # training loop

    for epoch in range (start_epoch, hp['num_epochs']):
        g_loss = 0
        d_loss = 0

        for i, img in enumerate(iterator):
            print("--------------------------------------------------")
            img = img.to(device)
            noise_input = torch.randn(hp['batch_size'], 100, 1, 1)
            noise_input = noise_input.to(device)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            gen_out = generator(noise_input)
            disc_real_out = discriminator(img.float())
            disc_fake_out = discriminator(gen_out)

            # loss and back prop

            # discriminator real loss
            target = torch.ones(hp['batch_size'], 1).to(device)
            d_loss_real = loss(disc_real_out, target)

            # discriminator fake loss
            target = torch.zeros(hp['batch_size'], 1).to(device)
            d_loss_fake = loss(disc_fake_out, target)

            disc_loss = d_loss_fake + d_loss_real

            disc_loss.backward(retain_graph=True)
            d_optimizer.step()

            # generator_loss
            target = torch.ones(hp['batch_size'], 1).to(device)
            gen_loss = loss(disc_fake_out, target)

            gen_loss.backward()
            g_optimizer.step()

            g_loss += gen_loss.item()
            d_loss += disc_loss.item()
            print("epoch = {}, Training_sample={}, generator_loss ={}".format(epoch, i, gen_loss))
            print("epoch = {}, Training_sample={}, discriminator_loss ={}".format(epoch, i, disc_loss))

        # average loss per epoch
        generator_loss[epoch] = g_loss/(i+1)
        discriminator_loss[epoch] = d_loss/(i+1)

        print("epoch = {}, average generator_loss ={}".format(epoch, generator_loss[epoch]))
        print("epoch = {}, average discriminator_loss ={}".format(epoch, discriminator_loss[epoch]))

        # plot loss curves and save model

        plt.plot(range(1, len(generator_loss)+1), generator_loss, 'b-', label="Generator loss")
        plt.xlabel("epochs")
        plt.ylabel("Error")
        plt.legend(loc='best')
        plt.savefig(checkpoints_folder + "/generator_loss.jpeg", bbox_inches="tight")
        plt.clf()

        plt.plot(range(1, len(discriminator_loss)+1), discriminator_loss, 'b-', label="Discriminator loss")
        plt.xlabel("epochs")
        plt.ylabel("Error")
        plt.legend(loc='best')
        plt.savefig(checkpoints_folder + "/discriminator_loss.jpeg", bbox_inches="tight")
        plt.clf()

        net_save = {'G': generator.state_dict(), 'D': discriminator.state_dict(), 'G_opt': g_optimizer.state_dict(),
                    'D_opt': d_optimizer.state_dict(), 'epoch': epoch}
        torch.save(net_save, checkpoints_folder + "/dc_gan_weights_epoch{}.pth".format(epoch))


if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    pre_trained_model_path = None  # provide path to .pth  to continue training from intermediate epochs
    train(pre_trained=pre_trained_model_path)

