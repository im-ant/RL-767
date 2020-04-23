# =============================================================================
# The surprising minizing agent, using Q learning
#
# Author: Anthony G. Chen
# =============================================================================

from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import network
import flat_replay_memory


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon_final):
    """
    TODO: organize stuff here and add start epsilon argument
    Code is copied largely directly from the Google Dopamine code

    :param decay_period: float, the period over which epsilon is decayed.
    :param step: int, the number of training steps completed so far.
    :param warmup_steps: number of steps taken before epsilon is decayed
    :param epsilon: the final epsilon value
    :return: current epsilon value
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon_final) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon_final)
    return epsilon_final + bonus


class SMiQFlatAgent(object):
    """
    The DQN Agent
    """

    def __init__(self, num_actions: int,
                 observation_shape: Tuple = (84,),
                 observation_dtype: torch.dtype = torch.uint8,
                 history_size: int = 4,
                 gamma: int = 0.9,
                 min_replay_history: int = 20000,
                 update_period: int = 4,
                 target_update_period: int = 8000,
                 epsilon_fn=linearly_decaying_epsilon,
                 epsilon_final: float = 0.1,
                 epsilon_decay_period: int = 250000,
                 memory_buffer_capacity: int = 1000000,
                 q_minibatch_size: int = 32,
                 vae_minibatch_size: int = 64,
                 seed: int = 42,
                 device: str = 'cpu'):
        """
        Initialize the Surprise minimizing Q-learning agent

        :param num_actions: number of actions the agent can take at any state.
        :param observation_shape: tuple of ints describing the observation
                                  shape, expects tuple of (n, )
        :param observation_dtype: NOTE: type of observaiton
        :param history_size: int, number of observations to use in state stack.
        :param gamma: decay constant
        :param min_replay_history: number of transitions that should be
            experienced before the agent begins training its value function
        :param update_period: int, number of actions between network training
        :param target_update_period: update period of target network (per action)
        :param epsilon_fn: epsilon decay function
        :param epsilon_start: exploration rate at start
        :param epsilon_final: final exploration rate
        :param epsilon_decay_period: length of the epsilon decay schedule
        :param memory_buffer_capacity: total capacity of the memory buffer for replay
        :param device: 'cuda' or 'cpu', depending on if 'cuda' is available
        :param summary_writer: TODO implement this with TensorBoard in the future
        """

        # ==
        # Set attributes

        # Environment related attributes
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype
        self.history_size = history_size
        self.gamma = gamma

        # Exploration related attributes
        self.epsilon_fn = epsilon_fn
        self.epsilon_final = epsilon_final
        self.epsilon_decay_period = epsilon_decay_period

        # Learning related attributes
        self.min_replay_history = min_replay_history
        self.update_period = update_period
        self.target_update_period = target_update_period
        self.memory_buffer_capacity = memory_buffer_capacity
        self.q_minibatch_size = q_minibatch_size
        self.q_lr = 0.00025

        # Density model related attributes
        self.vae_lr = 1e-3
        self.vae_minibatch_size = vae_minibatch_size
        self.vae_latent_dim = 32

        # System related attributes
        self.seed = seed
        self.device = device
        self.rng = np.random.RandomState(seed=self.seed)

        # ==
        # Initialize network, memory and optimizer

        # Policy, target networks and augmented state memory
        self.policy_net = None
        self.target_net = None
        self.pol_optimizer = None
        self.aug_s_buffer = None

        # Density estimate, and experience buffer
        self.vae = None
        self.vae_optimizer = None
        self.exp_buffer = None

        self._init_network()
        self._init_memory()

        # History queue: for stacking observations (np matrices in cpu)
        self.history_queue = deque(maxlen=self.history_size)

        # ==
        # Per-episode density estimate parameters
        # We want to track E[Z] and Var(Z) = E[X^2] - E[X]^2
        # To avoid float cancellation, we use Var(Z-K) = Var(Z), so instead
        # we track: E[(Z-K)^2], where K is a per-episode shift constant
        self.z_mu = torch.zeros((1, self.vae_latent_dim),
                                device=self.device,
                                requires_grad=False)  # E[Z]
        self.z_shift_const = torch.zeros((1, self.vae_latent_dim),
                                         device=self.device,
                                         requires_grad=False)  # K
        self.z_sq_shifted = torch.zeros((1, self.vae_latent_dim),
                                        device=self.device,
                                        requires_grad=False)  # E[(Z-K)^2]

        # ==
        # Counter variables
        self.total_actions_taken = 0  # for epsilon decay
        self.total_optim_steps = 0  # for target network updates
        self._latest_epsilon = 1.0
        self.per_episode_log = {
            't': 0,
            'cumulative_log_prob': 0.0,
            'Q_optim_steps': 0,
            'total_Q_loss': 0.0,
            'vae_optim_steps': 0,
            'total_dec_loss': 0.0,
            'total_kld_loss': 0.0
        }

        self.action = None  # action to be taken (selected at prev timestep)
        self._prev_observation = None

    def _init_network(self) -> None:
        """Initialize the small flat Q network"""
        # ==
        # Initialize Q networks and its optimizer
        # Mapping from Q: (1, aug_state_dim) -> (1, num_actions)

        # Compute the augmented state dimension, which is the normal state +
        # the per-episode mean and variance estimates
        aug_state_dim = (np.prod(self.observation_shape) * self.history_size +
                         (2 * self.vae_latent_dim))

        # Initialize networks
        self.policy_net = network.mlp_network(
            input_size=aug_state_dim,
            hidden_size=32,
            num_actions=self.num_actions
        ).to(self.device)
        self.target_net = network.mlp_network(
            input_size=aug_state_dim,
            hidden_size=32,
            num_actions=self.num_actions
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # set target net to evaluation mode

        # https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop
        # TODO change to Adam?
        self.pol_optimizer = optim.RMSprop(self.policy_net.parameters(),
                                           lr=self.q_lr,
                                           alpha=0.95,
                                           momentum=0.0,
                                           eps=0.00001,
                                           centered=True)

        # ==
        # Initialize the density estimation VAE network
        # Mapping from V: (1, state_dim) -> (1, state_dim)
        self.vae = network.BernoulliVae(
            input_dim=np.prod(self.observation_shape) * self.history_size,
            hidden_dim=64,
            latent_dim=self.vae_latent_dim,
            output_dim=np.prod(self.observation_shape) * self.history_size,
            device=self.device
        ).to(self.device)

        # Optimizer
        self.vae_optimizer = optim.Adam(self.vae.parameters(),
                                        lr=self.vae_lr)

    def _init_memory(self) -> None:
        """
        Initialize memory buffers
            aug_s_buffer: augmented state buffer for policy
            exp_buffer: experience buffer for density model
        """
        # ==
        # Augmented state buffer for policy training
        # will always have history 1 since each entry is a full augmented
        # state (full history of observation + sufficient statistics of
        # density model)
        aug_state_dim = (np.prod(self.observation_shape) * self.history_size +
                         (2 * self.vae_latent_dim))
        self.aug_s_buffer = flat_replay_memory.CircularFlatReplayBuffer(
            buffer_cap=self.memory_buffer_capacity,
            history=1,
            obs_shape=(aug_state_dim,),
            obs_dtype=self.observation_dtype,
            device=self.device
        )

        # ==
        # (Normal) experience buffer for density model training
        self.exp_buffer = flat_replay_memory.CircularFlatReplayBuffer(
            buffer_cap=self.memory_buffer_capacity,
            history=self.history_size,
            obs_shape=self.observation_shape,
            obs_dtype=self.observation_dtype,
            device=self.device
        )

    def begin_episode(self, observation: np.ndarray) -> int:
        """
        Start the episode
        :param observation: first observation, with env.observation_space.shape
        :return: first action (idx) to be taken
        """
        # ==
        # Initialize (zero-padded) history
        # NOTE: each entry in queue is a (self.observation_shape) tensor
        for _ in range(self.history_size):
            zero_pad_mat = torch.zeros(self.observation_shape,
                                       dtype=self.observation_dtype,
                                       device='cpu')
            self.history_queue.append(zero_pad_mat)
        # Add the first observation to the history
        obs_tensor = torch.tensor(observation,
                                  dtype=self.observation_dtype,
                                  device='cpu')
        self.history_queue.append(obs_tensor)

        # ==
        # Construct augmented state and pick action
        state_tensor = self._history_queue_to_state()  # (1, feature * history)
        aug_state_tensor = self._generate_aug_state(state_tensor)
        action = self._select_action(aug_state_tensor)

        # ==
        # Update density estimate
        with torch.no_grad():
            init_z, init_z_var = self.vae.encode(state_tensor)
        self.z_mu = init_z.clone()
        self.z_shift_const = init_z.clone()
        self.z_sq_shifted *= 0.0  # E[(Z-K)^2] = 0 with initial observation

        # ==
        # Reset per-episode counters
        for k in self.per_episode_log:
            self.per_episode_log[k] *= 0

        # ==
        # Return action
        self.action = action
        self._prev_observation = observation
        self.total_actions_taken += 1
        # TODO potential bug above that might skip some training steps?
        return self.action

    def step(self, observation: np.ndarray, reward: float, done: bool) -> int:
        """
        The agent takes one step

        :param observation: o_t, observation from environment, should
                            have shape self.observation_shape
        :param reward: r_t, float reward received
        :param done: done_t, bool, whether the episode is finished
        :return: int denoting action to take at this step
        """

        # ==
        # Store observation to history queue
        cur_obs_tensor = torch.tensor(observation,
                                      dtype=self.observation_dtype,
                                      device='cpu')
        self.history_queue.append(cur_obs_tensor)

        # ==
        # Store experience: o_{t-1}, a_{t-1}, r_t, done_t
        # First set them to the correct tensor, dtype and device
        obs_tensor = torch.tensor(self._prev_observation,
                                  dtype=self.observation_dtype,
                                  device=self.device)
        act_tensor = torch.tensor([self.action], dtype=torch.int32,
                                  device=self.device)
        rew_tensor = torch.tensor([reward], dtype=torch.float32,
                                  device=self.device)
        don_tensor = torch.tensor([done], dtype=torch.bool,
                                  device=self.device)
        self.exp_buffer.push(obs_tensor, act_tensor, rew_tensor, don_tensor)

        # ==
        # Store augmented state
        # Construct state and encode latent state
        state_tensor = self._history_queue_to_state()  # (1, feature * history)
        with torch.no_grad():
            z_tensor, z_var = self.vae.encode(state_tensor)

        # Generate augmented state (1, feature*history+2*latent) and rewards
        aug_s_tensor = self._generate_aug_state(state_tensor)
        aug_r_tensor = self._generate_aug_reward(z_tensor)  # (1, )

        self.aug_s_buffer.push(aug_s_tensor, act_tensor, aug_r_tensor,
                               don_tensor)

        # ==
        # Training step
        self._train_step()

        # ==
        # Select action
        action = self._select_action(aug_s_tensor)

        # ==
        # Update density estimates using running average
        t = self.per_episode_log['t'] + 1
        self.z_mu += (1 / (t + 1)) * (z_tensor - self.z_mu)  # E[Z]
        self.z_sq_shifted += ((1 / (t + 1)) *
                              ((z_tensor - self.z_shift_const).pow(2)
                               - self.z_sq_shifted))  # E[(Z-K)^2]

        # ==
        # Update counters and return
        self.action = action
        self._prev_observation = observation
        self.total_actions_taken += 1
        self.per_episode_log['t'] += 1
        self.per_episode_log['cumulative_log_prob'] += aug_r_tensor.item()

        return self.action

    def _history_queue_to_state(self) -> torch.tensor:
        """
        Convert the current history queue into a torch tensor state, where the
        state is sent to device (same as the neural nets).

        Each entry in the self.history_queue is a (self.observation_shape)
        tensor, which is concatenated at the first dimension, with an
        additional dimension added at the 0-oth position

        :return: torch.tensor object of state, of shape (1, feature * history)
        """
        state = torch.cat(list(self.history_queue), dim=0).unsqueeze(0)
        state = state.type(torch.float32).to(self.device)
        return state

    def _generate_aug_state(self, state_tensor):
        """
        Generate the "augmented state" for policy training, augmented state
        include the state and the sufficient statistics of the current
        density model (so it is fully observable for the policy to learn to
        maximize prob. of the current density model)

        :param state_tensor: torch tensor of shape (1, feature*history)
        :return: augmented state tensor of shape
                 (1, feature*history + 2*vae_latent_dim)
        """
        # Compute the variance: Var(Z) = E[(Z-K)^2] - (E[Z]-K)^2
        z_var = ((self.z_sq_shifted
                  - (self.z_mu - self.z_shift_const).pow(2))
                 + 1e-10)  # stability

        # Move state to device and concatenate with sufficient statistics
        state_tensor = state_tensor.to(self.device)
        aug_state = torch.cat((state_tensor, self.z_mu, z_var), dim=1)
        return aug_state

    def _generate_aug_reward(self, z_tensor):
        """
        Generate the augmented reward, based on the current (encoded)
        observation and the current episode density estimation

        :param obs_tensor: torch tensor of shape (1, feature)
        :return: "reward" tensor of shape (1,)
        """
        # Compute the variance: Var(Z) = E[(Z-K)^2] - (E[Z]-K)^2
        z_var = ((self.z_sq_shifted
                  - (self.z_mu - self.z_shift_const).pow(2))
                 + 1e-10)  # stability

        # Generate augmented reward
        aug_r = - torch.sum((
                (0.5 * z_var.log()) +
                ((z_tensor - self.z_mu).pow(2) /
                 (2 * z_var))
        ))

        return aug_r

    def _select_action(self, state) -> int:
        """
        Select action according to the epsilon greedy policy
        :param state: the AUGMENTED state
        :return: int action index
        """

        # Compute epsilon
        epsilon = self.epsilon_fn(self.epsilon_decay_period,
                                  self.total_actions_taken,
                                  self.min_replay_history,
                                  self.epsilon_final)
        self._latest_epsilon = epsilon

        # ===
        # Epsilon greedy policy
        if self.rng.uniform() <= epsilon:
            # random action with probability epsilon
            return self.rng.choice(self.num_actions)
        else:
            # greedy action
            with torch.no_grad():
                # Get values (1, n_actions), then take max column index
                action_tensor = self.policy_net(state).max(1)[1].view(1, 1)
            return action_tensor.item()

    def _train_step(self) -> None:
        """
        Run a singe training step. Train both the policy network and the
        density estimate. Occasionally update the target network
        """

        # ==
        # Optimize the policy network and Q estimates
        if len(self.aug_s_buffer) > self.min_replay_history:
            # Update policy network
            if self.total_actions_taken % self.update_period == 0:
                if len(self.aug_s_buffer) >= self.q_minibatch_size:
                    self._optimize_policy()
            # Update target network
            if self.total_optim_steps % self.target_update_period == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # ==
        # Optimize the density estimate
        if len(self.exp_buffer) >= self.vae_minibatch_size:
            self._optimize_density()

    def _optimize_policy(self) -> float:
        """
        Optimizes the policy (Q) network
        """
        # ==
        # Sample augmented states and unpack
        mem_batch = self.aug_s_buffer.sample(self.q_minibatch_size)
        (state_batch, action_batch, reward_batch,
         next_state_batch, done_batch) = mem_batch

        state_batch = state_batch.type(torch.float32).to(self.device)
        action_batch = action_batch.type(torch.long).to(self.device)
        reward_batch = reward_batch.type(torch.float32).to(self.device)
        next_state_batch = next_state_batch.type(torch.float32).to(self.device)
        done_batch = done_batch.type(torch.bool).to(self.device)

        # ==
        # Compute TD error

        # Get policy net output (batch, n_actions), extract action (index)
        # which need to have shape (batch, 1) for torch.gather to work.
        state_action_values = self.policy_net(state_batch) \
            .gather(1, action_batch)  # (batch-size, 1)

        # Get semi-gradient Q-learning targets (no grad to next state)
        next_state_values = self.target_net(next_state_batch) \
            .max(1)[0].unsqueeze(1).detach()  # (batch-size, 1)
        # Note that if episode is done do not use bootstrap estimate
        expected_state_action_values = (((next_state_values * (~done_batch))
                                         * self.gamma)
                                        + reward_batch)

        # Compute TD loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # ==
        # Optimization
        self.pol_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # gradient-clipping
            param.grad.data.clamp_(-1, 1)
        self.pol_optimizer.step()

        # Log the loss
        self.per_episode_log['total_Q_loss'] += loss.item()
        self.per_episode_log['Q_optim_steps'] += 1
        self.total_optim_steps += 1

    def _optimize_density(self):
        """
        Optimize the density estimate
        """
        # ==
        # Sample experience to get states and unpack
        mem_batch = self.exp_buffer.sample(self.vae_minibatch_size)
        (state_batch, action_batch, reward_batch,
         next_state_batch, done_batch) = mem_batch

        state_batch = state_batch.type(torch.float32).to(self.device)

        # ==
        # VAE training
        mu, log_var = self.vae.encode(state_batch)
        z_vec = self.vae.sample_z(mu, log_var)
        recon_batch = self.vae.decode(z_vec)

        # Loss
        dec_loss = F.binary_cross_entropy(recon_batch, state_batch,
                                          reduction='mean')
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        vae_loss = dec_loss + kld

        # Optimization
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # ==
        # Logging
        self.per_episode_log['total_dec_loss'] += dec_loss.item()
        self.per_episode_log['total_kld_loss'] += kld.item()
        self.per_episode_log['vae_optim_steps'] += 1

    def report(self, episode_idx, logger=None):
        # ==
        # Compute the averages

        # Per-episode average log probability
        avg_log_prob = (self.per_episode_log['cumulative_log_prob'] /
                        self.per_episode_log['t'])


        # Per-episode policy optimization loss
        avg_Q_loss = 0.0
        if self.per_episode_log['Q_optim_steps'] > 0:
            avg_Q_loss = (self.per_episode_log['total_Q_loss'] /
                          self.per_episode_log['Q_optim_steps'])

        # Per episode VAE logs
        avg_dec_loss = 0.0
        avg_kld_loss = 0.0
        if self.per_episode_log['vae_optim_steps'] > 0:
            avg_dec_loss = (self.per_episode_log['total_dec_loss'] /
                            self.per_episode_log['vae_optim_steps'])
            avg_kld_loss = (self.per_episode_log['total_kld_loss'] /
                            self.per_episode_log['vae_optim_steps'])


        # ==
        # Print or log
        if episode_idx == 0:
            print("\tEpsilon || total_actions || total_optims || avg_Q_loss")

        if logger is None:
            print(f"  {self._latest_epsilon} || "
                  f"{self.total_actions_taken} || "
                  f"{self.total_optim_steps} || "
                  f"{avg_Q_loss}")
        else:
            logger.add_scalar('Timesteps', self.per_episode_log['t'],
                              global_step=episode_idx)

            logger.add_scalar('Eps_exploration', self._latest_epsilon,
                              global_step=episode_idx)
            logger.add_scalar('Total_actions', self.total_actions_taken,
                              global_step=episode_idx)
            logger.add_scalar('Total_pol_optimizations', self.total_optim_steps,
                              global_step=episode_idx)

            logger.add_scalar('Per_episode_avg_Q_loss', avg_Q_loss,
                              global_step=episode_idx)
            logger.add_scalar('Per_episode_cumulative_log_prob',
                              self.per_episode_log['cumulative_log_prob'],
                              global_step=episode_idx)
            logger.add_scalar('Per_episode_avg_log_prob',
                              avg_log_prob,
                              global_step=episode_idx)

            logger.add_scalar('Per_episode_vae_optim_steps',
                              self.per_episode_log['vae_optim_steps'],
                              global_step=episode_idx)
            logger.add_scalar('Per_episode_avg_dec', avg_dec_loss,
                              global_step=episode_idx)
            logger.add_scalar('Per_episode_avg_kld', avg_kld_loss,
                              global_step=episode_idx)


if __name__ == "__main__":
    # for testing run this directly
    print('testing')
    agent = DQNAgent(num_actions=8)
    print(agent)
    print(agent.policy_net)
    print(agent.target_net)
    print(agent.memory.capacity)
