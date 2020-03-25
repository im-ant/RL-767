# ============================================================================
# Module implementing a replay memory buffer
#
# Some references:
#   - https://github.com/transedward/pytorch-dqn (which were taken from the
#       Berkeley deep RL course)
#   - Google dopamine (in tf, should cross-check against this at some point
#       to make sure the replication is exact):
#       https://github.com/google/dopamine/blob/master/dopamine/replay_memory/circular_replay_buffer.py
#
# Author: Anthony G. Chen
# ============================================================================

from typing import List

import numpy as np
import torch


class CircularReplayBuffer(object):
    """
    Circular replay buffer for naive uniform sampling of the recent past

    With default dtypes, the DQN (Mnih 2015) full implementation with 84*84
    frame size and 1mil frames should take around 7.1 GB to store
    """

    def __init__(self, buffer_cap=10000,
                 history=4,
                 obs_shape=(1, 84, 84),
                 obs_dtype=torch.uint8,
                 seed=2,
                 device='cpu') -> None:
        """
        Initializing the circular replay buffer. Assumes that the observation
        shape is always of shape (channel, feature 1, *); That is, at least
        2-dimensional (including channel dimension)

        :param buffer_cap: total capacity of the buffer
        :param history: number of observation to stack to make a state
        :param obs_shape: shape of observation tensor
        :param device: device to store memory on
        """

        # Initialize counter
        self._cur_idx = 0  # Current buffer index to write to
        self.size = 0  # Number of experiences stored
        self.capacity = buffer_cap  # Total buffer capacity
        self.history = history  # History length (# frames for a state)
        self._device = device  # Device (cpu/cuda) to store buffers on

        # Initialize the experience shapes and types
        self._obs_shape = obs_shape
        self._obs_dtype = obs_dtype
        self._act_dtype = torch.int32
        self._rew_dtype = torch.float32

        # Initialize the experience buffers
        obs_buffer_shape = ((self.capacity,) + self._obs_shape)
        self._obs_buffer = torch.empty(obs_buffer_shape, dtype=self._obs_dtype,
                                       device=self._device)
        self._act_buffer = torch.empty(self.capacity, dtype=self._act_dtype,
                                       device=self._device)
        self._rew_buffer = torch.empty(self.capacity, dtype=self._rew_dtype,
                                       device=self._device)
        self._done_buffer = torch.empty(self.capacity, dtype=torch.bool,
                                        device=self._device)

        self.rng = np.random.RandomState(seed=seed)

    def push(self, observ: torch.tensor, action: torch.tensor,
             reward: torch.tensor, done: torch.tensor) -> None:
        """
        Pushes an experience to the buffer
        """
        # Write experiences to buffer
        # TODO: add assertions to check for types?
        self._obs_buffer[self._cur_idx] = observ
        self._act_buffer[self._cur_idx] = action
        self._rew_buffer[self._cur_idx] = reward
        self._done_buffer[self._cur_idx] = done

        self._cur_idx = (self._cur_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) \
            -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        """
        Sample minibatch experience from the buffer
        :param n: batch size
        :return: batches of state, action, reward and next state
        """
        # Sample the last indeces (inclusive) of sequences to extract,
        #   sampling done inside of range [0, self.size)
        # TODO?: modify such one cannot sample index i where done[i-1]==True
        #   this is because if i is the last obs of successor state then the
        #   previous state is completely absent
        indeces = self.rng.choice(self.size, size=n, replace=False)

        # Get the buffered experiences
        state_batch, next_state_batch = self.encode_states(indeces)
        act_batch = self._act_buffer[indeces]
        rew_batch = self._rew_buffer[indeces]
        don_batch = self._done_buffer[indeces]

        return state_batch, act_batch, rew_batch, next_state_batch, don_batch

    def encode_states(self, idxs: np.ndarray) -> (torch.tensor, torch.tensor):
        """
        Generate the state (stacks of observations) given the indeces of the
        last observation, zero-padding if crossing episode boundaries
        :param idxs: indeces of the last observation of the successor state
        :return: minibatches of state and successor states
        """

        # Initialize current and next states tensors (pre zero-padded)
        # State shape: (batch size, channels*history, feature 1, *)
        _state_tensor_shape = ((len(idxs), self.history * self._obs_shape[0])
                               + self._obs_shape[1:])
        cur_states = torch.zeros(_state_tensor_shape,
                                 dtype=self._obs_dtype,
                                 device=self._device)
        nex_states = torch.zeros(_state_tensor_shape,
                                 dtype=self._obs_dtype,
                                 device=self._device)

        # Fill each state
        for i, buf_idx in enumerate(idxs):
            # Get valid obs sequence of length history + 1
            # NOTE: previously I had _get_valid_seq((buf_idx + 1) % self.size), not sure why
            seq_idxs = self._get_valid_seq(buf_idx)
            cur_seq_idxs = seq_idxs[:-1]
            nex_seq_idxs = seq_idxs[1:] if len(seq_idxs) > self.history \
                else seq_idxs  # TODO why do this if-else statement? I forget

            # Fill states
            # NOTE reshaping stacks multiple channels (if present) AND frames
            #      into the 2nd dimension of the returned state tensor
            if len(cur_seq_idxs) > 0:
                # number of (non zero-padded) observations & channels to stack
                non_zero_len = len(cur_seq_idxs) * self._obs_shape[0]
                # reshape from ( len(seq_idxs), channels, feature 1, * )
                #   to ( len(seq_idxs)*channels, feature 1, * )
                cur_states[i, -non_zero_len:] = torch.reshape(
                    self._obs_buffer[cur_seq_idxs],
                    ((non_zero_len,) + self._obs_shape[1:])
                )

            if len(nex_seq_idxs) > 0:
                non_zero_len = len(nex_seq_idxs) * self._obs_shape[0]
                nex_states[i, -non_zero_len:] = torch.reshape(
                    self._obs_buffer[nex_seq_idxs],
                    ((non_zero_len,) + self._obs_shape[1:])
                )

        return cur_states, nex_states

    def _get_valid_seq(self, last_idx: int) -> List[int]:
        """
        Helper method to return a sequence of valid (observation) indeces
        (inclusive) to form (both) the previous and successor states.
        Length range from be 1 to self.history+1

        :param last_idx: last index of the sequence
        :return: List of valid memory buffer indeces of length: # of valid
                    memory entries (can be shorter than self.history)
        """
        assert last_idx < self.size

        # Get the allowable first index of this sequence
        first_idx = last_idx
        for j in range(1, self.history + 1):
            cur_idx = last_idx - j
            # If it goes out of bound (index < 0)..
            if cur_idx < 0:
                # If buffer is full, then re-circle to end of buffer
                if self.size == self.capacity:
                    cur_idx = cur_idx % self.capacity
                # If buffer not full then terminate
                else:
                    break
            # If it crosses an episode boundary
            if self._done_buffer[cur_idx]:
                break
            # If all good then keep going
            first_idx = cur_idx

        # Get the sequence of indeces
        valid_seq = []
        while first_idx != ((last_idx + 1) % self.capacity):
            valid_seq.append(first_idx)
            first_idx = (first_idx + 1) % self.capacity

        return valid_seq

    def __len__(self) -> int:
        return self.size


def main():
    """For testing / debug purposes only """
    buf = CircularReplayBuffer(buffer_cap=15,
                               obs_shape=(1, 2, 1),
                               history=3
                               )

    for _ob in range(20):
        cur_obs = torch.ones((1, 2, 1),
                             dtype=torch.float32) * (_ob + 1)
        is_done = False
        if (_ob + 1) % 8 == 0:
            is_done = True

        buf.push(cur_obs, 1, 2, is_done)

        print(f'Buffer size: {len(buf)}; Capacity: {buf.capacity}')

    print('\nbuffer._done_buffer:')
    print(buf._done_buffer)

    print('\nValid sequence ending at idx 9')
    print(buf._get_valid_seq(9))
    print('\nValid sequence ending at idx 0')
    print(buf._get_valid_seq(0))

    print('\nGenerate states ending at 0,8')
    cur_s, next_s = buf.encode_states([0, 8])
    print(f'cur_s size: {cur_s.size()}; next_s size: {next_s.size()}')

    print('At index 0 (channel*history, feature 1):')
    print('\tCurrent state', cur_s[0, :, :, 0])
    print('\tNext state', next_s[0, :, :, 0])

    print('At index 8:')
    print('\tCurrent state', cur_s[1, :, :, 0])
    print('\tNext state', next_s[1, :, :, 0])

    print('\nSampling 15 examples')
    s, a, r, sp = buf.sample(15)
    print('state size', s.size())
    print('action', a)
    print('reward', r)
    print('next state size', sp.size())


if __name__ == "__main__":
    main()
