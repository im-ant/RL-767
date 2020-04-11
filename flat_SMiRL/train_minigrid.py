# ============================================================================
# Training set-up for the minigrid environment
#
# Uses tensorboard for logging:
# https://pytorch.org/docs/stable/tensorboard.html
# ============================================================================

import argparse

import gym
import numpy as np

from gym_minigrid import wrappers
import torch
from torch.utils.tensorboard import SummaryWriter

import classic_agent
import smi_flat_agent

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================================
# Helper methods for organizing observation spaces
# ========================================
def get_observation_shape(env):
    print(env.envs[0].unwrapped.spec.id)
    pass

def get_flat_observation(obs_dict):
    pass

def minigrid_flat_states(obs_dict):
    """
    (Hacky) helper method to get a dictionary of observation from MiniGrid
    to a flattened vector form. Specifically, the dictionary should contain
    the key value pairs, which is flattened and concatenated:
    "image": 7x7x3 np.ndarray with a compact coding of the surrounding,
             partially observed environment
    "direction": direction agent is facing

    :param obs_dict: dictionary of observation from MiniGrid environment
    :return: flattened np.ndarray observation of shape (148,)
    """
    # Flatten the 7x7x3 (partially observable) surround
    flat_obs = obs_dict['image'].flatten()
    # Concatenate the direction
    #agent_dir = obs_dict['direction']
    #flat_obs = np.concatenate((flat_obs, [agent_dir]))
    # TODO maybe keep maybe delete

    return flat_obs


# ========================================
# Helper methods for agent initialization
# ========================================

def init_agent(args, env, logger=None):
    # NOTE: observation shape is set to (148,) for the "flattened" MiniGrid
    # environment (7x7x3) compact image encoding and 1 for facing direction

    AGENT = 'fnn'

    if AGENT is 'linear':
        agent = classic_agent.LinearQAgent(n_actions=env.action_space.n,
                                           obs_shape=(148,),
                                           gamma=0.9,
                                           epsilon=0.1,
                                           lr=0.001)
    if AGENT is 'smi_fc':
        agent = smi_flat_agent.FnnQAgent(n_actions=env.action_space.n,
                                         obs_shape=(148,),
                                         gamma=0.95,
                                         seed=args.seed,
                                         device=DEVICE)
    else:
        agent = classic_agent.FnnQAgent(n_actions=env.action_space.n,
                                        obs_shape=(147,), # TODO change 148 -> 147 for all
                                        gamma=0.95,
                                        seed=args.seed,
                                        device=DEVICE)

    return agent


def run_environment(args: argparse.Namespace):
    # Logging
    if args.log_dir is not None:
        logger = SummaryWriter(log_dir=args.log_dir)
    else:
        logger = None

    # TODO: set seed for environment

    # Create environment and agent
    env = gym.make(args.env_name)
    if args.log_dir is not None:
        agent = init_agent(args, env, logger)
    else:
        agent = init_agent(args, env, None)


    print(f'Starting training, {args.num_episode} episodes')
    for epis_idx in range(args.num_episode):
        # Reset environment and agent
        observation = env.reset()
        observation = minigrid_flat_states(observation)
        action = agent.begin_episode(observation)

        # Counters
        cumu_reward = 0.0
        timestep = 0

        while True:
            # Iteract
            observation, reward, done, info = env.step(action)
            observation = minigrid_flat_states(observation)
            action = agent.step(observation, reward, done)

            cumu_reward += reward
            timestep += 1

            if done:
                # Logging
                if args.log_dir is not None:
                    logger.add_scalar('Reward', cumu_reward,
                                      global_step=epis_idx)
                    if epis_idx % 10 == 0:
                        print(f'Epis {epis_idx}, Timesteps: {timestep}, Return: {cumu_reward}')

                else:
                    print(f'Epis {epis_idx}, Timesteps: {timestep}, Return: {cumu_reward}')

                # Agent logging TODO: not sure if this is the best practice
                agent.report(logger=logger, epis_idx=epis_idx)
                break

            # TODO: have some debugging print-out (e.g. every 100 episode) to make sure times and
            # things are good and training is happening

    env.close()
    if args.log_dir is not None:
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training on MiniGrid environment')

    parser.add_argument('--num_episode', type=int, default=20, metavar='N',
                        help='number of episodes to run the environment for (default: 500)')
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-8x8-v0', metavar='N',
                        help='environment name (default: MiniGrid-Empty-8x8-v0)')
    parser.add_argument('--log_dir', type=str, default=None, metavar='',
                        help='Path to the log output directory (default: ./log_dir/)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='Seed for rng (default: 0)')

    # Parse
    print(DEVICE)
    args = parser.parse_args()
    print(args)

    run_environment(args)
