# ============================================================================
# Agent set-up for surprising-minimizing agents with flat observations
#
# In general, the agent interfaces with the world in either standard python
# or numpy data types, and cast to torch tensors as needed.
# ============================================================================

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import network
from replay_memory import CircularReplayBuffer


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon_final):
    """
    TODO: organize stuff here and add start epsilon argument
    Code is copied largely directly from the Google Dopamine code
    :param decay_period: float, the period over which epsilon is decayed.
    :param step: int, the number of training steps completed so far.
    :param warmup_steps: number of steps taken before epsilon is decayed
    :param epsilon_final: the final epsilon value
    :return: current epsilon value
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon_final) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon_final)
    return epsilon_final + bonus


class FnnQAgent(object):
    """
    Q-learning agent with fully connected neural net
    """

    def __init__(self, n_actions: int,
                 obs_shape: Tuple,
                 gamma=0.9,
                 seed=2,
                 device='cpu'):
        # Environment attributes
        self.n_actions = n_actions
        self.obs_shape = obs_shape

        # MDP related attributes
        self.gamma = gamma
        self.epsilon_final = 0.01
        self.decay_period = 10000
        self.warmup_steps = 10

        # Q network parameters
        self.q_net_lr = 0.001
        self.q_minibatch_size = 32
        self.target_update_steps = 64

        # Memory buffer parameters
        self.buffer_capacity = 1000

        # VAE parameters
        self.vae_lr = 1e-3
        self.vae_minibatch_size = 64
        self.vae_latent_dim = 32

        #
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        self.device = device

        # Memory and network initialization
        self.aS_buffer = None
        self.policy_net = None
        self.target_net = None
        self.pol_optimizer = None

        # VAE initialization
        self.ex_buffer = None
        self.vae = None
        self.vae_optimizer = None

        self._init()

        # ==
        # Logging variables
        self.prev_action = None

        self.total_steps = 0
        self.total_Q_training_steps = 0

        self.per_episode_log = {
            't': 0,
            'cumulative_log_prob': 0.0,
            'Q_optim_steps': 0,
            'total_Q_loss': 0.0,
            'vae_optim_steps': 0,
            'total_dec_loss': 0.0,
            'total_kld_loss': 0.0
        }

        # Per-episode density model parameters TODO why per-episode?
        self.z_shift = torch.zeros((1, self.vae_latent_dim),  # for num stabil
                                   device=self.device,
                                   requires_grad=False)
        self.z_mu = torch.zeros((1, self.vae_latent_dim),
                                device=self.device,
                                requires_grad=False)
        self.z_shifted_mu2 = torch.zeros((1, self.vae_latent_dim),
                                         device=self.device,
                                         requires_grad=False)

        # TODO use tensorboard.add_custom_scalars(layout)
        # https://pytorch.org/docs/stable/tensorboard.html
        # To add hyperparameters used

    def _init(self):
        """Temporary function to initialize things"""
        # Augmented state dimension
        # Observation dimension + 2 * latent state dimension
        aug_state_shape = ((np.prod(self.obs_shape) + (2 * self.vae_latent_dim)),)

        # ===
        # Buffers
        # Experience buffer of past observations (hacky, since I only need to
        # store observations but I'm storing s,a,r,s)
        self.ex_buffer = CircularReplayBuffer(buffer_cap=self.buffer_capacity,
                                              history=1,
                                              obs_shape=((1,) + self.obs_shape),
                                              obs_dtype=torch.float32,
                                              seed=self.seed * 3,
                                              device=self.device)
        # Augmented state buffer for policy training
        self.aS_buffer = CircularReplayBuffer(buffer_cap=self.buffer_capacity,
                                              history=1,
                                              obs_shape=((1,) + aug_state_shape),
                                              obs_dtype=torch.float32,
                                              seed=self.seed * 3,
                                              device=self.device)

        # ===
        # Policy network
        # Init policy and target networks
        # TODO: need to also pass in the sufficient statistic from the density estimation
        self.policy_net = network.mlp_network(np.prod(aug_state_shape), 24,
                                              self.n_actions).to(self.device)
        self.target_net = network.mlp_network(np.prod(aug_state_shape), 24,
                                              self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # set target net to evaluation mode

        # Optimizer
        self.pol_optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=self.q_net_lr,
                                        betas=(0.9, 0.999),  # rest is default
                                        eps=1e-08, weight_decay=0,
                                        amsgrad=False)

        # ===
        # VAE
        self.vae = network.GaussianVae(input_dim=np.prod(self.obs_shape),
                                       hidden_dim=64,
                                       latent_dim=self.vae_latent_dim,
                                       output_dim=np.prod(self.obs_shape),
                                       device=self.device,
                                       ).to(self.device)

        self.vae_optimizer = optim.Adam(self.vae.parameters(),
                                        lr=self.vae_lr,
                                        betas=(0.9, 0.999),  # rest is default
                                        eps=1e-08, weight_decay=0,
                                        amsgrad=False)

    def begin_episode(self, observation: np.ndarray):
        # Cast to torch tensor of size (channel, feature 1, *)
        # Note below is specifically for "flat" observations (e.g. (4,))
        obs_tensor = np.reshape(observation, (1, len(observation)))  # (1, obs)
        obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32)

        # Generate augmented states (1, obs + 2*vae_latent_dim)
        aug_s_tensor = self._generate_aug_state(obs_tensor)

        # ==
        # Pick action
        action = self._select_action(aug_s_tensor)
        self.prev_action = action

        # ==
        # Reset variables
        for k in self.per_episode_log:
            self.per_episode_log[k] *= 0

        # ==
        # Initialize variables for mu and variance estimation
        obs_tensor = obs_tensor.to(self.device)
        with torch.no_grad():
            obs_z, obs_z_var = self.vae.encode(obs_tensor)
        self.z_shift = obs_z.clone()
        self.z_mu = obs_z.clone()
        self.z_shifted_mu2 *= 0.0  # shifted to zero

        return action

    def step(self, observation: np.ndarray, reward: float, done: bool) -> int:
        """
        Observe the environment and take an action

        :param observation:
        :param reward:
        :param done:
        :return: int indexing the action to be taken
        """
        # ==
        # Type cast the input and store to experience memory

        # Cast to torch tensor of size (channel, feature 1, *)
        # Note below is specifically for "flat" observations (e.g. (4,))
        obs_tensor = np.reshape(observation, (1, len(observation)))
        obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32).clone()

        # Cast reward and action
        act_tensor = torch.tensor(self.prev_action)
        rew_tensor = torch.tensor(reward).clone()
        don_tensor = torch.tensor(done).clone()

        # Store to experience buffer
        self.ex_buffer.push(obs_tensor, act_tensor, rew_tensor, don_tensor)

        # ==
        # Store augmented state
        # Encode to latent space
        obs_tensor = obs_tensor.to(self.device)
        with torch.no_grad():
            obs_z, obs_z_var = self.vae.encode(obs_tensor)
        # Generate augmented state and rewards
        aug_s_tensor = self._generate_aug_state(obs_tensor)
        aug_r_tensor = self._generate_aug_reward(obs_z)

        # Store to augmented state buffer
        self.aS_buffer.push(aug_s_tensor, act_tensor, aug_r_tensor, don_tensor)

        # ==
        # Training step
        self._train_step()

        # ==
        # Select action
        action = self._select_action(aug_s_tensor)
        self.prev_action = action

        # ==
        # Update estimate
        self.per_episode_log['t'] += 1
        t = self.per_episode_log['t']
        self.z_mu += (1 / (t + 1)) * (obs_z - self.z_mu)
        self.z_shifted_mu2 += ((1 / (t + 1)) *
                               ((obs_z - self.z_shift).pow(2) -
                                self.z_shifted_mu2))

        # Track reward estimate
        self.per_episode_log['cumulative_log_prob'] += aug_r_tensor.item()

        return action

    def _generate_aug_state(self, obs_tensor):
        """
        Generate the "augmented state" for policy training, augmented state
        include the observation and the sufficient statistics of the current
        density model (so it is fully observable for the policy to learn to
        maximize prob. of the current density model)

        :param obs_tensor: torch tensor of shape (1, feature)
        :return: augmented state tensor of shape (1, feature + 2*vae_latent_dim)
        """
        # Compute the variance
        z_var = ((self.z_shifted_mu2 - (self.z_mu - self.z_shift).pow(2))
                 + 1e-10)

        obs_tensor = obs_tensor.to(self.device)
        aug_s = torch.cat((obs_tensor, self.z_mu, z_var), dim=1)
        return aug_s

    def _generate_aug_reward(self, z_tensor):
        """
        Generate the augmented reward, based on the current (encoded)
        observation and the current episode density estimation

        :param obs_tensor: torch tensor of shape (1, feature)
        :return: "reward" tensor of shape (1,)
        """
        # Compute the variance
        z_var = ((self.z_shifted_mu2 - (self.z_mu - self.z_shift).pow(2))
                 + 1e-10)

        # Generate augmented reward
        aug_r = - torch.sum((
                (0.5 * z_var.log()) +
                ((z_tensor - self.z_mu).pow(2) /
                 (2 * z_var))
        ))

        return aug_r

    def _train_step(self) -> None:
        """
        Take one step training, evaluate whether or not to optimize model
        depending on memory buffer size, and whether to update the
        parameters of the target network
        :return: None
        """

        # ==
        # Optimize the policy network Q estimates
        if len(self.aS_buffer) >= self.q_minibatch_size:
            self._optimize_policy()

            # Update target network with policy network
            if self.total_Q_training_steps % self.target_update_steps == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.total_Q_training_steps += 1

        # ==
        # Optimize the density model
        if len(self.ex_buffer) >= self.vae_minibatch_size:
            self._optimize_density()

        # ==
        # Increment counter
        self.total_steps += 1

    def _sample_format_memory(self, batch_size, buffer) -> Tuple:
        """
        Helper function to sample the memory buffer, format the minibatch of
        sampled data in the right type, shape and device, and return

        :return: Tuple containing:
            state_batch: (batch size, feature dim)
            action_batch: (batch size, )    TODO confirm this
            reward_batch: (batch size, 1)
            next_state_batch: (batch size, feature dim)
            done_batch: (batch size, 1)
        """
        # ==
        # Sample memory and unpack to the right shapes
        mem_batch = buffer.sample(batch_size)
        state_batch, action_batch, reward_batch, \
        next_state_batch, done_batch = mem_batch

        state_batch = state_batch.type(torch.float32) \
            .view((batch_size, -1)).to(self.device)  # (batch, feat)

        action_batch = action_batch.type(torch.long).to(self.device)

        reward_batch = reward_batch.type(torch.float32) \
            .view((batch_size, -1)).to(self.device)

        next_state_batch = next_state_batch.type(torch.float) \
            .view((batch_size, -1)).to(self.device)  # (batch, feat)

        done_batch = done_batch.view((batch_size, -1)) \
            .to(self.device)  # (batch, 1)

        out_tup = (state_batch, action_batch, reward_batch,
                   next_state_batch, done_batch)

        return out_tup

    def _optimize_density(self) -> None:
        """
        Optimize the density estimation
        :return:
        """

        # ==
        # Sample
        mem_batch = self._sample_format_memory(self.vae_minibatch_size,
                                               self.ex_buffer)
        (state_batch, action_batch, reward_batch,
         next_state_batch, done_batch) = mem_batch

        # ==
        # VAE training
        mu, log_var = self.vae.encode(state_batch)
        z_vec = self.vae.sample_z(mu, log_var)
        recon_batch = self.vae.decode(z_vec)

        # Loss
        # Note: binary decoder has BCE loss, here we use Gaussian + fix var
        dec_loss = F.mse_loss(recon_batch, state_batch, reduction='mean')
        # dec_loss = F.binary_cross_entropy(recon_batch, state_batch,
        #                                   reduction='mean')  # TODO remove BCE loss
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        vae_loss = dec_loss + kld

        # Gradient update
        vae_loss.backward()
        self.vae_optimizer.step()

        # ==
        # Logging loss
        self.per_episode_log['total_dec_loss'] += dec_loss.item()
        self.per_episode_log['total_kld_loss'] += kld.item()
        self.per_episode_log['vae_optim_steps'] += 1

    def _optimize_policy(self) -> None:
        """
        Optimizes the policy net model
        :return:
        """

        # ==
        # Sample
        mem_batch = self._sample_format_memory(self.q_minibatch_size,
                                               self.aS_buffer)
        (state_batch, action_batch, reward_batch,
         next_state_batch, done_batch) = mem_batch

        # ==
        # Compute target (current & expected values)

        # Get policy net output (batch, n_actions), extract with action index
        # which needs to have shape (batch, 1) for torch.gather to work.
        # this gets us the value of action taken
        state_action_values = self.policy_net(state_batch) \
            .gather(1, action_batch.view(-1, 1))  # (batch-size, 1)

        # Get semi-gradient Q-learning targets
        # (Note the next state value is detached from compute graph)
        next_state_values = self.target_net(next_state_batch) \
            .max(1)[0].unsqueeze(1).detach()  # (batch-size, 1)
        # Note that if episode is done do not use bootstrap estimate
        expected_state_action_values = (((next_state_values * (~done_batch))
                                         * self.gamma)
                                        + reward_batch)

        # Compute TD loss (DQN uses smooth_l1_loss I believe)
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # ==
        # Optimization
        self.pol_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # gradient-clipping
            param.grad.data.clamp_(-1, 1)
        self.pol_optimizer.step()

        # ==
        # Logging the loss
        self.per_episode_log['total_Q_loss'] += loss.item()
        self.per_episode_log['Q_optim_steps'] += 1

    def _select_action(self, s_tensor: torch.Tensor):
        """
        Epsilon-greedy policy

        :param s_tensor: torch tensor of size (channel, feature 1, *)
        :return:
        """

        eps = linearly_decaying_epsilon(decay_period=self.decay_period,
                                        step=self.total_steps,
                                        warmup_steps=self.warmup_steps,
                                        epsilon_final=self.epsilon_final)

        # Ensure input has size (1, feature) for MLP
        s_tensor = s_tensor.clone().detach().view(1, -1).to(self.device)

        # ==
        if self.rng.uniform() <= eps:
            return self.rng.choice(self.n_actions)
        else:
            with torch.no_grad():
                # Get values (1, n_actions), then take max column index
                action = self.policy_net(s_tensor).max(1)[1] \
                    .view(1).item()
                return action

    def report(self, logger=None, epis_idx=None):
        # ==
        # Compute averages

        # Per episode Q-learning logs
        avg_policy_loss = 0.0
        if self.per_episode_log['Q_optim_steps'] > 0:
            avg_policy_loss = (self.per_episode_log['total_Q_loss'] /
                               self.per_episode_log['Q_optim_steps'])

        # Per episode VAE logs
        avg_dec_loss = 0.0
        avg_kld_loss = 0.0
        if self.per_episode_log['vae_optim_steps'] > 0:
            avg_dec_loss = (self.per_episode_log['total_dec_loss'] /
                            self.per_episode_log['vae_optim_steps'])
            avg_kld_loss = (self.per_episode_log['total_kld_loss'] /
                            self.per_episode_log['vae_optim_steps'])

        # Epsilon exploration rate
        eps = linearly_decaying_epsilon(decay_period=self.decay_period,
                                        step=self.total_steps,
                                        warmup_steps=self.warmup_steps,
                                        epsilon_final=self.epsilon_final)

        # ==
        # Print or log
        if logger is None:
            print(f'Total steps: {self.total_steps}, '
                  f'Total Q training steps: {self.total_Q_training_steps}')
            print(f"\tEpis_cumu_log_prob: {self.per_episode_log['cumulative_log_prob']}\n"
                  f"\tEpis_Q_optim_steps: {self.per_episode_log['Q_optim_steps']}\n"
                  f"\tEpis_avg_Q_loss: {avg_policy_loss}\n"
                  f"\tEpis_vae_optim_steps_loss: {self.per_episode_log['vae_optim_steps']}\n"
                  f"\tEpis_avg_dec_loss: {avg_dec_loss}\n"
                  f"\tEpis_avg_kld_loss: {avg_kld_loss}\n")

        else:
            # Total steps and policy training steps
            logger.add_scalar('Total_steps', self.total_steps,
                              global_step=epis_idx)
            logger.add_scalar('Total_Q_training_steps', self.total_Q_training_steps,
                              global_step=epis_idx)

            logger.add_scalar('Eps_exploration', eps,
                              global_step=epis_idx)

            logger.add_scalar('Per_episode_cumulative_log_prob',
                              self.per_episode_log['cumulative_log_prob'],
                              global_step=epis_idx)

            logger.add_scalar('Per_episode_Q_optim_steps',
                              self.per_episode_log['Q_optim_steps'],
                              global_step=epis_idx)
            logger.add_scalar('Per_episode_avg_Q_loss', avg_policy_loss,
                              global_step=epis_idx)

            logger.add_scalar('Per_episode_vae_optim_steps',
                              self.per_episode_log['vae_optim_steps'],
                              global_step=epis_idx)
            logger.add_scalar('Per_episode_avg_dec', avg_dec_loss,
                              global_step=epis_idx)
            logger.add_scalar('Per_episode_avg_kld', avg_kld_loss,
                              global_step=epis_idx)


if __name__ == "__main__":
    agent = FnnQAgent(n_actions=3,
                      obs_shape=(4,), )

    print(agent.buffer)
    print(agent.policy_net)
    print(agent.target_net)
    print()

    obs = np.array([0.0, 0.1, 0.2, 0.3])
    reward = 1.0
    done = False

    agent.begin_episode(obs)
    for i in range(30):
        action = agent.step(obs, reward, done)
        print(f'epis{i}, action: {action}')
        if agent.cur_epis_optim_steps > 0:
            print(agent.cur_epis_optim_steps, agent.cur_epis_policy_loss / agent.cur_epis_optim_steps)
