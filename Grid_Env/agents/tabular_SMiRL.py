# ============================================================================
# Tabular agent doing model-based surprise minimization
# ============================================================================

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TabularMinSurpriseAgent(object):
    """
    Agent to minimize surprise
    """

    def __init__(self,
                 n_states,
                 gamma=0.9,
                 state_prior=None,
                 trans_mat=None,
                 k_samples=10,
                 trajectory_len=10,
                 seed=2):

        # Initialization
        self.n_states = n_states
        self.n_action = 4
        self.gamma = gamma

        # Action sampling
        self.k_samples = k_samples
        self.trajectory_len = trajectory_len

        # Rng
        self.rng = np.random.RandomState(seed=seed)

        # ==
        # Initialize state space (optionally with a prior)
        if state_prior is not None:
            self.state_count = state_prior
        else:
            self.state_count = np.ones(self.n_states, dtype=np.int32)
            # Note the state count is initialized with laplace smoothing
        # Normalize counts for the density model
        self.density = self.state_count / np.sum(self.state_count)

        # ==
        # Initialize transition model (optionally given the perfect model)
        if trans_mat is not None:
            self.trans_count = None
            self.trans_prob = trans_mat
        else:
            self.trans_count = np.ones(
                (self.n_states, self.n_action, self.n_states),
                dtype=np.int32
            )
            self.trans_prob = None

        # ==
        # Tracking variables
        self.prev_observation = None
        self.action = None

        self.per_episode_log = {
            't': 0,
            'total_log_prob': 0.0
        }

    def begin_episode(self, obs):
        """
        :param obs: assumed to be the state index
        """
        # Reset variables
        for k in self.per_episode_log:
            self.per_episode_log[k] *= 0

        # Increment state count and density model
        self.state_count[obs] += 1
        self.density = self.state_count / np.sum(self.state_count)

        # Update variables and choose action
        self.prev_observation = obs
        self.action = self._select_action(obs)
        return self.action

    def step(self, obs, reward, done):
        """
        :param obs: assumed to be the state index
        """
        # ==
        # Update state density if not done
        if not done:
            self.state_count[obs] += 1
            self.density = self.state_count / np.sum(self.state_count)

        # ==
        # Update transition
        if self.trans_count is not None:
            self.trans_count[self.prev_observation, self.action, obs] += 1

        # ==
        # Per episode tracking
        self.per_episode_log['t'] += 1
        self.per_episode_log['total_log_prob'] += np.log(self.density[obs])

        # ==
        # Update variables and select actions
        self.prev_observation = obs
        self.action = self._select_action(obs)
        return self.action

    def _select_action(self, obs):
        """
        :param obs: assumed to be the state index
        """

        # ==
        # Normalize to get the transition probabilities
        if self.trans_count is not None:
            t_rowsum = np.sum(self.trans_count, axis=2, keepdims=True)
            trans = self.trans_count / t_rowsum
        else:
            trans = self.trans_prob

        # ==
        # Sample trajectories via random shooting

        # Uniformly sample K action trajectories
        a_samps = self.rng.choice(self.n_action, replace=True,
                                  size=(self.k_samples, self.trajectory_len))
        # Track returns
        sampled_G = np.zeros(self.k_samples, dtype=np.float32)
        # Apply the samples to the learned dynamics
        for sample_i in range(self.k_samples):
            cur_s = obs
            for traj_t in range(self.trajectory_len):
                # Uniformly pick action
                cur_a = a_samps[sample_i, traj_t]
                # Sample next state
                nex_s = self.rng.choice(self.n_states, p=trans[cur_s, cur_a])

                # TODO: there might be an error with the trans here when the transition or
                # state density is given?

                # Accumulate reward
                cur_discount = self.gamma ** traj_t
                sampled_G[sample_i] += cur_discount * np.log(self.density[nex_s])
                # Update
                cur_s = nex_s

        # ==
        # Pick the first action with the highest return
        best_traj_idx = np.argmax(sampled_G)
        return a_samps[best_traj_idx, 0]

    def report(self, logger=None, episode_idx=None):
        """
        Write to log
        :param logger: torch tensorboard SummaryWriter
        :param episode_idx: int denoting episode index
        """

        if logger is None:
            return None
        else:
            logger.add_scalar('Epis_total_log_prob',
                              self.per_episode_log['total_log_prob'],
                              global_step=episode_idx)





