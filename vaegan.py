import torch
import torch.nn as nn
import torch.nn.functional as F

bottleneck_size = 2 # n means, n log variances
drop = 0


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(784, 512)
        self.lin2 = nn.Linear(512, 128)
        self.lin3 = nn.Linear(128, 64)
        self.mean_head = nn.Linear(64, bottleneck_size)
        self.std_head  = nn.Linear(64, bottleneck_size)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.dropout(self.lin1(x), p=drop, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=drop, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin3(x), p=drop, training=self.training)
        x = F.relu(x)
        return self.mean_head(x), self.std_head(x)

    def isTraining(self, flag):
        for param in self.parameters():
            param.requires_grad = flag

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(bottleneck_size, 64)
        self.lin2 = nn.Linear(64, 128)
        self.lin3 = nn.Linear(128, 512)
        self.lin4 = nn.Linear(512, 784)
        self.sig  = nn.Sigmoid()

    def forward(self, z):
        x = F.dropout(self.lin1(z), p=drop, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=drop, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin3(x), p=drop, training=self.training)
        x = F.relu(x)
        return self.sig(self.lin4(x))

    def isTraining(self, flag):
        for param in self.parameters():
            param.requires_grad = flag

class Discriminator(nn.Module):
    # For the discriminator, we not only want to
    # give a prediction as to whether the image is real (1)
    # or fake (0), but also give image features from an intermediate layer

    def __init__(self):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(784, 256)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, x): # Given 784 either real or fake
        x = F.dropout(self.lin1(x), p=drop, training=self.training)
        features_1 = F.relu(x)

        x = F.dropout(self.lin2(features_1), p=drop, training=self.training)
        features_2 = F.relu(x)

        y_hat = self.sig(self.lin3(features_2))
        return y_hat, features_1, features_2

    def isTraining(self, flag):
        for param in self.parameters():
            param.requires_grad = flag







'''
class AdversarialVAE(nn.Module):

    def __init__(self):
        super(AdversarialVAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.mean_head = nn.Linear(64, bottleneck_size)
        self.std_head  = nn.Linear(64, bottleneck_size)

        # Decoder
        self.d_fc1 = nn.Linear(bottleneck_size, 64)
        self.d_fc2 = nn.Linear(64, 128)
        self.d_fc3 = nn.Linear(128, 512)
        self.d_fc4 = nn.Linear(512, 784)
        self.d_sig = nn.Sigmoid()

    def forward_encoder(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.mean_head(x), self.std_head(x)

    # Backprop can't flow through a random node.
    # So this wouldn't work if we sampled from a gaussian with mean mu and std e^0.5logvar
    # But instead, we can sample from a gaussian to get a z-score, then use that to create our sample
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        rand_z_score = torch.randn_like(std)
        return mu + rand_z_score*std

    def forward_decoder(self, z):
        x = F.relu(self.d_fc1(z))
        x = F.relu(self.d_fc2(x))
        x = F.relu(self.d_fc3(x))
        x = self.d_fc4(x)
        x = self.d_sig(x)
        return x

    def forward(self, x):
       mu, logvar = self.forward_encoder(x)
       z = self.sample(mu, logvar)
       x_prime = self.forward_decoder(z)
       return x_prime, mu, logvar # Also pass latent dim forward so we can calculate loss all in one place
'''
# end
