# VAEGAN implementation in Pytorch - with added interactive visualization
This repository is a simple MNIST implementation of "Autoencoding beyond pixels with a learned similarity metric".

The adversarial (perceptual loss: content and style) objective subsumes invariances such as rotation, translation, etc. Whereas pixel-wise distance between an image and a small shifted image would be very large, perceptual distance (feature-wise distance) would not, and this is more in tune with what a human expects.

Goals: 
1. Hierarchical/image compositional object embeddings
2. Explain/show the effects of adversarial training
3. Improve training of GANs ala SPADE
	a. Perceptual Photoshop ala SPADE
5. Improve computational understanding of images
6. Beat SOTA for USS --> Make it easy to make segmentation datasets


Notes:
1. We jump-start the training process by using element-wise reconstruction loss for a small number of epochs at the beginning
2. We only train the discriminator if its loss is above a certain value
3. We only use a bottleneck size of 2 so that we can use the interactive visualization. This is a pretty bad idea for model performance, however.
4. Latent space differences:
	a. In the plain autoencoder, classes were clustered chaotically and close together
	b. In the variational autoencoder, clusters were long lines, and were separated by small distances
	c. In this autoencoder with adversarial perceptual loss, the latent space is clustered still with mostly clean separations, but also clusters are small blobs instead of long lines.
5. Total trainable params: 
	a. 1,170,325
	


## How to run
1. Run `python3 run.py` to train the model (this creates `model.pt` in the `checkpoints` folder.
2. Run `python3 visualize.py` to see the latent space for the model you just trained.

There are also pretrained weights available, so you can just do step 2 if you want.
