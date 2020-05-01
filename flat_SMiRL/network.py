# =============================================================================
# The DQN network architecture
#
# Very much inspired by / adopted from:
# https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_output_length(input_length: int, kernel_size: int,
                          stride: int) -> int:
    """
    Compute the output (side) length of a conv2d operation

    :param input_length: length of the input 2d matrix along this dimension
    :param kernel_size: size of the kernel along this dimension
    :param stride: stride of the kernel along this dimension
    :return: int denoting the length of output
    """
    out_length = input_length - (kernel_size - 1) - 1
    out_length = out_length / stride + 1
    return int(out_length)


class nature_dqn_network(nn.Module):
    """
    The Mnih 2015 Nature DQN network, as described in
    https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning
    """

    def __init__(self, num_actions: int, num_channels: int,
                 input_shape: Tuple = (84, 84)):
        """

        :param num_actions: number of allowable actions
        :param num_channels: number of input image channels (i.e. number of
                stacked frames)
        """
        super(nature_dqn_network, self).__init__()

        self.num_actions = num_actions
        self.num_channels = num_channels
        self.input_shape = input_shape

        # Initialize conv layers
        self.conv1 = nn.Conv2d(self.num_channels, 32,
                               kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)

        # Compute num of units: https://pytorch.org/docs/stable/nn.html#conv2d
        # also github.com/transedward/pytorch-dqn/blob/master/dqn_model.py
        # NOTE: assume square kernel
        side_length = compute_output_length(input_shape[0], 8, 4)
        side_length = compute_output_length(side_length, 4, 2)
        side_length = compute_output_length(side_length, 3, 1)

        # Initialize fully connected layers; num units computed from:
        self.fc1 = nn.Linear(64 * side_length * side_length, 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h = h.view(h.shape[0], -1)  # flatten convolution
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class mlp_network(nn.Module):
    """
    Simple feed-forward network with 2 hidden layers
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_actions: int):
        """
        :param input_size: size of input
        :param num_actions: number of allowable actions
        """
        super(mlp_network, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # Initialize layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

        # TODO maybe add dropout / regularization

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class BernoulliVae(nn.Module):
    """
    Feed-forward VAE
    """

    # Constructor
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 output_dim: int,
                 device: str = 'cpu'):
        super(BernoulliVae, self).__init__()

        # Attributes
        self.input_d = input_dim
        self.hidden_d = hidden_dim
        self.latent_d = latent_dim
        self.output_d = output_dim

        self.device = device

        # Encoder layers
        self.fc1 = nn.Linear(self.input_d, self.hidden_d)
        self.mu = nn.Linear(self.hidden_d, self.latent_d)
        self.log_var = nn.Linear(self.hidden_d, self.latent_d)

        # Decoding layers
        self.fc2 = nn.Linear(self.latent_d, self.hidden_d)
        self.output = nn.Linear(self.hidden_d, self.output_d)

    # Encoder
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mean = self.mu(h)
        log_variance = self.log_var(h)  # log var for easier computation later

        return mean, log_variance

    # Sampler
    def sample_z(self, mean, log_variance):
        eps = torch.randn(self.latent_d,
                          device=self.device)
        z = mean + (torch.exp(log_variance / 2) * eps)

        return z

    # Decoder
    def decode(self, z):
        h = F.relu(self.fc2(z))
        y = torch.sigmoid(self.output(h))
        # y = self.output(h)  # for fixed variance Gaussian decode

        return y

    # Forward method
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.sample_z(mean, log_var)
        y = self.decode(z)
        return y



if __name__ == "__main__":
    # for testing run this directly
    print('testing')
    # test_dqn = nature_dqn_network(8, 64)
    # print(test_dqn)

    test_fc = mlp_network(4, 64, 3)
    print(test_fc)
