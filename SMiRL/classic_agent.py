# ============================================================================
# Agent set-up for classical RL agents
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

from network import mlp_network
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


class LinearQAgent(object):
    """
    Q-learning agent with linear approximation
    """

    def __init__(self,
                 n_actions: int,
                 obs_shape: Tuple = (4,),
                 gamma=0.9,
                 epsilon=0.1,
                 lr=0.1,
                 seed=2):
        # Environment attributes
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.phi_shape = np.prod(obs_shape)  # feature shape

        # MDP related attributes
        self.gamma = gamma
        self.epsilon = epsilon  # TODO set a schedule

        self.rng = np.random.RandomState(seed=seed)

        # Weight matrix to compute each action independently
        # NOTE that this assumes a flat input feature vector
        self.W = self.rng.uniform(-0.0001, 0.0001,
                                  size=(self.phi_shape, self.n_actions))
        self.lr = lr

        # Keeping track of previous state
        self.prev_phi = None
        self.prev_act = None

    def _basis_fn(self, obs: np.ndarray) -> np.ndarray:
        """
        Takes in a observation and transform it into a feature

        :param obs: observation vector, of size self.obs_shape
        :return: feature vector, of size self.phi_shape
        """
        return obs

    def begin_episode(self, observation: np.ndarray):
        # Reset and get random actin
        self.prev_phi = None
        self.prev_act = None
        action = self.step(reward=0.0, observation=observation, done=False)

        # Log and return
        self.prev_phi = self._basis_fn(observation)
        self.prev_act = action
        return action

    def step(self, observation: np.ndarray, reward: float, done: bool) -> int:

        # ==
        # Value computation
        phi = self._basis_fn(observation)  # size (phi_shape,)
        cur_Q = np.dot(phi, self.W)  # size (n_actions,)

        # ==
        # Learning
        if self.prev_phi is not None:
            # Value estimate for previous state
            prev_Q = np.dot(self.prev_phi, self.W)
            prev_q = prev_Q[self.prev_act]

            # Target value
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(cur_Q)

            # TD error and update
            td_err = target_q - prev_q
            self.W[:, self.prev_act] += self.lr * td_err * self.prev_phi

        # ==
        # Policy
        if self.rng.uniform() < self.epsilon:
            action = self.rng.choice(self.n_actions)
        else:
            action = np.argmax(cur_Q)

        # Log and return
        self.prev_phi = phi
        self.prev_act = action
        return action

    def report(self, logger, epis_idx):
        pass


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
        self.decay_period = 5000
        self.warmup_steps = 10

        #
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        self.device = device

        # Memory and network initialization
        self.buffer = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        self._init()

        # Neural net parameters
        self.minibatch_size = 32
        self.target_update_steps = 64

        # Logging variables
        self.prev_action = None
        self.cur_epis_policy_loss = 0.0
        self.cur_epis_optim_steps = 0
        self.training_steps = 0
        self.total_steps = 0


        # TODO use tensorboard.add_custom_scalars(layout)
        # https://pytorch.org/docs/stable/tensorboard.html
        # To add hyperparameters used


    def _init(self):
        """Temporary function to initialize things"""
        # Initialize memory buffer
        self.buffer = CircularReplayBuffer(buffer_cap=1000,
                                           history=1,
                                           obs_shape=((1,) + self.obs_shape),
                                           obs_dtype=torch.float32,
                                           seed=self.seed * 3,
                                           device=self.device)

        # Init networks
        self.policy_net = mlp_network(np.prod(self.obs_shape), 24,
                                      self.n_actions).to(self.device)
        self.target_net = mlp_network(np.prod(self.obs_shape), 24,
                                      self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # set target net to evaluation mode

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=0.001,
                                    betas=(0.9, 0.999),  # rest is default
                                    eps=1e-08, weight_decay=0,
                                    amsgrad=False)

    def begin_episode(self, observation: np.ndarray):
        # Cast to torch tensor of size (channel, feature 1, *)
        # Note below is specifically for "flat" observations (e.g. (4,))
        obs_tensor = np.reshape(observation, (1, len(observation)))  # (1, obs)
        obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32)

        # Pick action
        action = self._select_action(obs_tensor)

        # Update or reset variables
        self.prev_action = action
        self.cur_epis_policy_loss = 0.0
        self.cur_epis_optim_steps = 0

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
        # Type cast the input and store to memory

        # Cast to torch tensor of size (channel, feature 1, *)
        # Note below is specifically for "flat" observations (e.g. (4,))
        obs_tensor = np.reshape(observation, (1, len(observation)))
        obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32).clone()

        # Cast reward and action
        act_tensor = torch.tensor(self.prev_action)
        rew_tensor = torch.tensor(reward).clone()
        don_tensor = torch.tensor(done).clone()

        # Store to memory
        self.buffer.push(obs_tensor, act_tensor, rew_tensor, don_tensor)

        # ==
        # Training step
        self._train_step()
        self.total_steps += 1

        # ==
        # Select action
        action = self._select_action(obs_tensor)
        self.prev_action = action
        return action

    def _train_step(self) -> None:
        """
        Take one step training, evaluate whether or not to optimize model
        depending on memory buffer size, and whether to update the
        parameters of the target network
        :return: None
        """
        if len(self.buffer) >= self.minibatch_size:
            self._optimize_model()

            if self.training_steps % self.target_update_steps == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.training_steps += 1

    def _optimize_model(self) -> None:
        """
        Optimizes the policy net model
        :return:
        """
        # If not enough memory
        assert len(self.buffer) >= self.minibatch_size

        # ==
        # Sample memory and unpack to the right shapes
        mem_batch = self.buffer.sample(self.minibatch_size)
        state_batch, action_batch, reward_batch, \
            next_state_batch, done_batch = mem_batch

        state_batch = state_batch.type(torch.float32) \
            .view((self.minibatch_size, -1)).to(self.device)  # (batch, feat)
        action_batch = action_batch.type(torch.long).to(self.device)
        reward_batch = reward_batch.type(torch.float32) \
            .view((self.minibatch_size, -1)).to(self.device)
        next_state_batch = next_state_batch.type(torch.float) \
            .view((self.minibatch_size, -1)).to(self.device)  # (batch, feat)
        done_batch = done_batch.view((self.minibatch_size, -1)) \
            .to(self.device)  # (batch, 1)


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
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # gradient-clipping
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # ==
        # Logging the loss
        self.cur_epis_policy_loss += loss.item()
        self.cur_epis_optim_steps += 1

    def _select_action(self, obs_tensor: torch.Tensor):
        """
        Epsilon-greedy policy

        :param obs_tensor: torch tensor of size (channel, feature 1, *)
        :return:
        """

        eps = linearly_decaying_epsilon(decay_period=self.decay_period,
                                        step=self.total_steps,
                                        warmup_steps=self.warmup_steps,
                                        epsilon_final=self.epsilon_final)

        # Ensure input has size (1, feature) for MLP
        obs_tensor = obs_tensor.clone().detach().view(1, -1).to(self.device)

        # ==
        if self.rng.uniform() <= eps:
            return self.rng.choice(self.n_actions)
        else:
            with torch.no_grad():
                # Get values (1, n_actions), then take max column index
                action = self.policy_net(obs_tensor).max(1)[1] \
                    .view(1).item()
                return action

    def report(self, logger=None, epis_idx=None):
        if logger is None:
            print(f'Total steps: {self.total_steps}, '
                  f'Training steps: {self.training_steps}')
            print(f'Cur epis avg policy loss: {self.cur_epis_policy_loss}, '
                  f'with {self.cur_epis_optim_steps} optimization steps')
        else:
            logger.add_scalar('Total_steps', self.total_steps,
                              global_step=epis_idx)
            logger.add_scalar('Total_train_steps', self.training_steps,
                              global_step=epis_idx)

            # Include the current epsilon exploration rate
            eps = linearly_decaying_epsilon(decay_period=self.decay_period,
                                            step=self.total_steps,
                                            warmup_steps=self.warmup_steps,
                                            epsilon_final=self.epsilon_final)
            logger.add_scalar('Eps_exploration', eps,
                              global_step=epis_idx)

            # Add the (policy net) optimization loss
            if self.cur_epis_optim_steps > 0:
                avg_policy_loss = self.cur_epis_policy_loss / self.cur_epis_optim_steps
            else:
                avg_policy_loss = 0.0
            logger.add_scalar('Epis_optimization_steps', self.cur_epis_optim_steps,
                              global_step=epis_idx)
            logger.add_scalar('Epis_avg_policy_loss', avg_policy_loss,
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
