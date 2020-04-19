# ============================================================================
# "SlipperSlope" Grid World environment
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class SlipperySlopeGridWorld(gym.Env):
    """
    Slippery slope world where agent must fight against the slope
    by taking action, else it will slip into the lava

    Grid cell index reference:
        0: platform (not slippery)
        1: slope (slippery)
        2: lava (death)

    Action index reference:
        0: left
        1: up
        2: right
        3: down
    """

    def __init__(self,
                 width=20,
                 lava_width=6,
                 slip_prob=0.3,
                 init_coords=[(5, 0)],
                 seed=2):
        # ==
        # World attributes
        self.N = width  # width of the world
        self.lava_width = lava_width
        self.slip_prob = slip_prob

        # Assertions
        assert self.N > self.lava_width

        # Action space
        self.action_space = spaces.Discrete(4)

        # ==
        # Construct the world
        self.grid = None
        self._init_grid()

        # ==
        # Grip: how probable you are to move in the intended direction
        self.grip_dict = {0: 0.999, 1: (1 - self.slip_prob), 2: 0}

        # ==
        # Map between coordinates and state index
        self.coord2idx = {}
        self.idx2coord = {}
        for y in range(self.N):
            for x in range(self.N):
                s_idx = y * self.N + x
                self.coord2idx[(y, x)] = s_idx
                self.idx2coord[s_idx] = (y, x)

        # ==
        # Transition probabilities
        self.Trans = None
        self._init_transition()

        # Other Env set-up
        self.observation_space = None
        self.rng = np.random.RandomState(seed=seed)
        self.initial_coords = init_coords

        self.state = None

    def _init_grid(self):
        """
        Helper method for initializing the grid world
        """
        # Overall grid
        grid = np.empty(shape=(self.N, self.N),
                        dtype=np.int32)
        # Lay the slope and platform
        grid[:, :] = 1
        grid[:, 0] = 0

        # Lay the lava
        grid[:, -self.lava_width:] = 2

        self.grid = grid

    def _init_transition(self):
        """
        Helper method for initializing the transition matrix
        Dictionary of mapping:
          T[s_1] -> {
            a1: ([s2, s3, ...],
                 [p_2, p_3, ...]),
            a2: ...
          }
        """
        # Initailize the dictionary with init state and actions
        T = {k: {l: None for l in range(self.action_space.n)} \
             for k in range(self.N ** 2)}

        # Iterate over all initial state & action pairs
        for s1_idx in T:
            s1_coord = self.idx2coord[s1_idx]
            for a in T[s1_idx]:
                # ==
                # Construct subsequent states from current (s,a) pair
                in_lava = (self.grid[s1_coord[0], s1_coord[1]] == 2)

                # If starting state is lava, go back to itself
                if in_lava:
                    s2_coord_list = [tuple(np.copy(s1_coord))]
                    s2_p_list = [1.0]

                # Else if not in lava state, construct possible subsequent states
                else:
                    s2_coord_list = []
                    s2_p_list = []
                    grip_prob = self.grip_dict[self.grid[s1_coord[0], s1_coord[1]]]
                    slip_prob = 1.0 - grip_prob

                    # Slip towards lava state
                    s2_slip_coord = np.copy(s1_coord)
                    if s1_coord[1] < (self.N - 1):
                        s2_slip_coord[1] = (s2_slip_coord[1] + 1)
                    s2_coord_list.append(tuple(s2_slip_coord))
                    s2_p_list.append(slip_prob)

                    # Move towards action taken
                    # If move taken is also in direction of lava
                    if a == 2:
                        s2_p_list[0] = 1.0
                    # Else if move taken is not towards lava
                    else:
                        s2_act_coord = np.copy(s1_coord)
                        if a == 0 and s1_coord[1] > 0:  # left no wall
                            s2_act_coord[1] = s2_act_coord[1] - 1
                        if a == 0 and s1_coord[1] == 0:  # left with wall reflection
                            s2_act_coord[1] = s2_act_coord[1] + 1
                        if a == 1:  # up
                            s2_act_coord[0] = (s2_act_coord[0] - 1) % self.N
                        if a == 3:  # down
                            s2_act_coord[0] = (s2_act_coord[0] + 1) % self.N
                        s2_coord_list.append(tuple(s2_act_coord))
                        s2_p_list.append(grip_prob)

                # TODO: construct so that if you go into the walls you'll be reflected
                # to some nearby states?

                # ==
                # Make s2 coord into index
                s2_idx_list = [self.coord2idx[c] for c in s2_coord_list]

                # Add this to the dictionary
                T[s1_idx][a] = (s2_idx_list, s2_p_list)

        self.Trans = T

    def get_transition_matrix(self):
        """
        Compute and return the transition matrix

        Takes the transition dictionry, self.Trans, and returns a
        transition matrix with shape (n_states, n_actions, n_states)
        """
        # Template for the matrix initailized at zero
        n_states = self.N * self.N
        n_action = self.action_space.n
        tMat = np.zeros((n_states, n_action, n_states), dtype=np.float32)

        # Iterate over s1, a, s2 and get its probability
        for s1 in self.Trans:
            for a in self.Trans[s1]:
                (s2_list, p_list) = self.Trans[s1][a]
                for i in range(len(s2_list)):
                    s2 = s2_list[i]
                    p = p_list[i]

                    tMat[s1, a, s2] = p

        return tMat

    def reset(self):
        init_coord_idx = self.rng.choice(len(self.initial_coords))
        self.state = self.coord2idx[self.initial_coords[init_coord_idx]]
        return self.state

    def step(self, action):
        """
        Step in the environment

        Action reference:
          0: left
          1: up
          2: right
          3: down
        """
        # Get next state
        (next_s, s_probs) = self.Trans[self.state][action]
        s_prime = np.random.choice(next_s, p=s_probs)

        # Terminal state?
        s_p_coord = self.idx2coord[s_prime]
        s_type = self.grid[s_p_coord[0], s_p_coord[1]]
        done = (s_type == 2)

        # Other info
        reward = int(done) * (-1)
        info = {}

        self.state = s_prime
        return s_prime, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def __str__(self):
        return "this is what gets printed"

    def _plot(self):
        """Helper method for visualizing the grid"""
        cmap = colors.ListedColormap(['green',
                                      'lightblue',
                                      'coral'])
        plt.imshow(self.grid, cmap=cmap)
