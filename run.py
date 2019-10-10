import time
import torch
import torchvision

import os.path
import numpy as np

import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from vaegan import Encoder, Decoder, Discriminator
##################################

batch_size = 128
batch_size_test = 1000
num_epochs = 1000

###################################
# Image loading and preprocessing
###################################

trainLoader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('./data', train=True, download=True,
		transform=torchvision.transforms.ToTensor()),
		# Usually would do a normalize, but for some reason this messes up the output
	batch_size=batch_size, shuffle=True)

testLoader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('./data', train=False, download=True,
		transform=torchvision.transforms.ToTensor()),
	batch_size=batch_size_test, shuffle=True)


#######################
# Model Setup
#######################

enc  = Encoder().cuda()
dec  = Decoder().cuda()
disc = Discriminator().cuda()
enc_optimizer  = torch.optim.Adam(enc.parameters(), lr=1e-3)
ae_optimizer   = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-3)

def enumerate_params():
	num_params = 0
	for model in [enc, dec, disc]:
		for param in model.parameters():
			if param.requires_grad:
				num_params += param.numel()

	print(f"Total trainable model parameters: {num_params}")
enumerate_params()


# We are using a Sigmoid layer at the end so we must use CE loss. Why?
# ---> Rather, paper said to use CE loss.
def reconstruction_loss(x, x_prime):
	binary_cross_entropy = F.binary_cross_entropy(x_prime, x, reduction='sum')
	return binary_cross_entropy

def kl_loss(mu, logvar):
	distance_from_standard_normal = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return distance_from_standard_normal

mse = nn.MSELoss()

def sample(mu, logvar):
    std = torch.exp(0.5*logvar)
    rand_z_score = torch.randn_like(std)
    return mu + rand_z_score*std

############################
### Actual Training Loop ###
############################

def ae_step(data):
	x, _ = data # Each 'data' is an image, label pair
	x = Variable(x).cuda() # Input image must be a tensor and moved to the GPU
	ae_optimizer.zero_grad()

	# Forward pass
	mu, logvar = enc(x)
	z = sample(mu, logvar)
	x_prime = dec(z)

	x_ = x.view(-1, 28*28)

	_, features_1_real, features_2_real = disc(x_)
	_, features_1_fake, features_2_fake = disc(x_prime)

	l_reconstruction = reconstruction_loss(x_, x_prime)
	l_kl             = kl_loss(mu, logvar)
	l_perceptual = mse(features_1_real, features_1_fake) + mse(features_2_real, features_2_fake)

	# Encoder/Decoder backward loss step
	#loss = (l_reconstruction+l_kl+l_perceptual)
	loss = (l_kl+l_perceptual)
	if reconstruct_epochs_left > 0:
		loss += l_reconstruction
	loss.backward(retain_graph=True)
	ae_optimizer.step()

	return l_reconstruction, l_kl, l_perceptual


def disc_step(data):
	x, _ = data # Each 'data' is an image, label pair
	x = Variable(x).cuda() # Input image must be a tensor and moved to the GPU
	disc_optimizer.zero_grad()
	# Forward pass
	mu, logvar = enc(x)
	z = sample(mu, logvar)
	x_prime = dec(z)

	x_ = x.view(-1, 28*28)

	disc_y_hat_real, features_1_real, features_2_real = disc(x_)
	disc_y_hat_fake, features_1_fake, features_2_fake = disc(x_prime)

	disc_y_hat_real = disc_y_hat_real.view(disc_y_hat_real.shape[0])
	disc_y_hat_fake = disc_y_hat_fake.view(disc_y_hat_fake.shape[0])

	ones  = torch.ones(disc_y_hat_real.shape[0]).cuda()
	zeros = torch.zeros(disc_y_hat_fake.shape[0]).cuda()

	l_disc    = mse(disc_y_hat_real, ones) + mse(disc_y_hat_fake, zeros)
	#l_ae_fake = mse(disc_y_hat_fake, ones) # Autoencoder should construct images that fool the disc
	# But this is better expressed by perceptual loss

	# Discriminator step
	l_disc.backward()
	disc_optimizer.step()

	percent_real_pred_real = np.mean(disc_y_hat_real.cpu().detach().numpy() > 0.5)
	percent_fake_pred_fake = np.mean(disc_y_hat_fake.cpu().detach().numpy() < 0.5)

	return l_disc, percent_real_pred_real, percent_fake_pred_fake

def get_disc_loss(data):
	x, _ = data
	x = Variable(x).cuda() # Input image must be a tensor and moved to the GPU

	mu, logvar = enc(x)
	z = sample(mu, logvar)
	x_prime = dec(z)

	x_ = x.view(-1, 28*28)

	disc_y_hat_real, features_1_real, features_2_real = disc(x_)
	disc_y_hat_fake, features_1_fake, features_2_fake = disc(x_prime)

	disc_y_hat_real = disc_y_hat_real.view(disc_y_hat_real.shape[0])
	disc_y_hat_fake = disc_y_hat_fake.view(disc_y_hat_fake.shape[0])

	ones  = torch.ones(disc_y_hat_real.shape[0]).cuda()
	zeros = torch.zeros(disc_y_hat_fake.shape[0]).cuda()

	percent_real_pred_real = np.mean(disc_y_hat_real.cpu().detach().numpy() > 0.5)
	percent_fake_pred_fake = np.mean(disc_y_hat_fake.cpu().detach().numpy() < 0.5)

	l_disc = mse(disc_y_hat_real, ones) + mse(disc_y_hat_fake, zeros)
	return l_disc


disc_wait = 8

reconstruct_epochs_left = 2 # How many epochs of reconstruction loss we use




i = 0
for epoch in range(num_epochs):
	# TrainLoader is a generator
	start = time.time()
	i = 0
	for data in trainLoader:
		i += 1
		if i % disc_wait == 0: # ae goes more times than disc
			if get_disc_loss(data) > 0.4:
				l_disc, percent_real_pred_real, percent_fake_pred_fake = disc_step(data)
			else:
				l_reconstruction, l_kl, l_perceptual = ae_step(data)
		else:
			l_reconstruction, l_kl, l_perceptual = ae_step(data)
	if epoch % 20 == 0:
		torch.save(enc, './checkpoints/enc.pt')
		torch.save(dec, './checkpoints/dec.pt')
		torch.save(disc, './checkpoints/disc.pt')




	elapsed = time.time() - start
	try:
		print('epoch [{}/{}], l_recon:{:.4f}, l_kl:{:.4f}, l_disc:{:.4f}, l_percept:{:.4f}, time:{:.2f}, r:{:.4f}, f:{:.4f}'.format(
			epoch+1, num_epochs, l_reconstruction.data, l_kl.data,
			l_disc.data, l_perceptual.data, elapsed,
			percent_real_pred_real, percent_fake_pred_fake))
	except:
		print("=============== Problem Encountered")

torch.save(enc, './checkpoints/enc.pt')
torch.save(dec, './checkpoints/dec.pt')
torch.save(disc, './checkpoints/disc.pt')


#######################
# Testing
#######################

images, labels = iter(testLoader).next()
images = Variable(images).cuda()
mu, logvar = enc(images)
z = sample(mu, logvar)
reconstructions = dec(z)
reconstructions = reconstructions.view(-1, 1, 28, 28)


# Display images / reconstructions
from matplotlib import pyplot as plt
def show(image):
	plt.imshow(image.permute(1, 2, 0))
	plt.show()

def show10(images1, images2):
	f, axes = plt.subplots(10, 2)
	for i in range(10):
		axes[i,0].imshow(images1.numpy()[i][0], cmap='gray')
		axes[i,1].imshow(images2.numpy()[i][0], cmap='gray')
	plt.show()

x  = images
x_ = reconstructions

show10(x.cpu(), x_.cpu().detach())
