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

from adversarialVAE import Encoder, Decoder, Discriminator
##################################

batch_size = 128
batch_size_test = 1000
num_epochs = 50

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
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(disc.parameters()), lr=1e-3)

# We are using a Sigmoid layer at the end so we must use CE loss. Why?
# ---> Rather, paper said to use CE loss.
def reconstruction_loss(x, x_prime):
	binary_cross_entropy = F.binary_cross_entropy(x_prime, x, reduction='sum')
	return binary_cross_entropy

def kl_loss(mu, logvar):
	distance_from_standard_normal = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return distance_from_standard_normal

def sample(mu, logvar):
    std = torch.exp(0.5*logvar)
    rand_z_score = torch.randn_like(std)
    return mu + rand_z_score*std

def lossFun(x, x_prime, mu, logvar):
	binary_cross_entropy = F.binary_cross_entropy(x_prime, x, reduction='sum')

	distance_from_standard_normal = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	# KL Divergence between the learned distribution and Normal(0, 1)

	return binary_cross_entropy + distance_from_standard_normal

############################
### Actual Training Loop ###
############################

for epoch in range(num_epochs):
	# TrainLoader is a generator
	start = time.time()
	for data in trainLoader:
		x, _ = data # Each 'data' is an image, label pair
		x = Variable(x).cuda() # Input image must be a tensor and moved to the GPU
		optimizer.zero_grad()

		# Forward pass
		mu, logvar = enc(x)
		z = sample(mu, logvar)
		x_prime = dec(z)
		predicted_authenticity = disc(x_prime)

		x = x.view(-1, 28*28)
		l_reconstruction = reconstruction_loss(x, x_prime)
		l_kl             = kl_loss(mu, logvar)

		loss = lossFun(x, x_prime, mu, logvar)
		loss.backward()
		# Backward pass
		#loss = (l_reconstruction+l_kl)
		#loss.backward()
		optimizer.step()

	elapsed = time.time() - start
	print('epoch [{}/{}], r_loss:{:.4f}, kl_loss:{:.4f} time:{:.2f}'.format(epoch+1, num_epochs, l_reconstruction.data, l_kl.data, elapsed))

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
