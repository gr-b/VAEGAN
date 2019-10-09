import numpy as np
import torch
import os.path
import torchvision
from matplotlib import pyplot as plt
from torch.autograd import Variable

###################################
#Image loading and preprocessing
###################################

batch_size_test = 1000

testLoader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('./data', train=False, download=True,
		transform=torchvision.transforms.ToTensor()),
	batch_size=batch_size_test, shuffle=True)

##############################
# Interactive Plot
##############################


if os.path.exists('./checkpoints/enc.pt') and os.path.exists('./checkpoints/dec.pt') and os.path.exists('./checkpoints/disc.pt') :
	print("Found models! Loading...")
else:
	print("No models found in checkpoints folder. Exiting.")
	exit()

enc  = torch.load('./checkpoints/enc.pt')
dec  = torch.load('./checkpoints/dec.pt')
disc = torch.load('./checkpoints/disc.pt')


images, labels = iter(testLoader).next()
images = Variable(images).cuda()

encoded_means, encoded_logvars = enc(images)
encoded_means, encoded_logvars = encoded_means.detach().cpu(), encoded_logvars.detach().cpu()

sizes = np.exp(0.5*encoded_logvars[:,0])*100*2

fig, ax = plt.subplots(1, 2)
ax[0].scatter(encoded_means[:,0], encoded_means[:,1],
	c=labels, s=sizes, cmap='tab10')

def onclick(event):
	global flag
	ix, iy = event.xdata, event.ydata
	try:
		latent_vec = torch.tensor([ix, iy])
	except:
		return
	latent_vec = Variable(latent_vec, requires_grad=False).cuda()

	decoded_img = dec(latent_vec)
	decoded_img = decoded_img.detach().cpu().numpy().reshape(28, 28)

	ax[1].imshow(decoded_img, cmap='gray')
	plt.draw()

cid = fig.canvas.mpl_connect('motion_notify_event', onclick)
plt.show()
