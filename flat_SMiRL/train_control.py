# ============================================================================
# Training set-up
#
# Uses tensorboard for logging:
# https://pytorch.org/docs/stable/tensorboard.html
# ============================================================================

import argparse

import gym
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import classic_agent
import smi_flat_agent

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_agent(args, env, logger=None):
    AGENT = 'smi_fc'

    if AGENT is 'linear':
        agent = classic_agent.LinearQAgent(n_actions=env.action_space.n,
                                           obs_shape=env.observation_space.shape,
                                           gamma=0.9,
                                           epsilon=0.1,
                                           lr=0.001)
    if AGENT is 'smi_fc':
        agent = smi_flat_agent.FnnQAgent(n_actions=env.action_space.n,
                                         obs_shape=env.observation_space.shape,
                                         gamma=0.95,
                                         seed=args.seed,
                                         device=DEVICE)
    else:
        agent = classic_agent.FnnQAgent(n_actions=env.action_space.n,
                                        obs_shape=env.observation_space.shape,
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

    # TODO set seed for environment

    # Initialize enviroment and agent
    env = gym.make(args.env_name)
    if args.log_dir is not None:
        agent = init_agent(args, env, logger)
    else:
        agent = init_agent(args, env, None)


    print(f'Starting training, {args.num_episode} episodes')
    for epis_idx in range(args.num_episode):

        # Reset environment and agent
        observation = env.reset()
        action = agent.begin_episode(observation)

        # Counters
        cumu_reward = 0.0
        timestep = 0

        while True:
            # Iteract
            observation, reward, done, info = env.step(action)
            action = agent.step(observation, reward, done)

            # NOTE hacky modification for cartpole
            # (maybe) just have it extend for a few episodes past end of cartpole?
            if done:
                reward = 0.0
                # Extend the run to at least 150 steps for cartpole TODO remove ths super hacky
                if timestep < 50:
                    done = False

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
    parser = argparse.ArgumentParser(description='DQN for atari environment')

    parser.add_argument('--num_episode', type=int, default=20, metavar='N',
                        help='number of episodes to run the environment for (default: 500)')
    parser.add_argument('--env_name', type=str, default='CartPole-v0', metavar='N',
                        help='environment name (default: CartPole-v0')
    parser.add_argument('--log_dir', type=str, default=None, metavar='',
                        help='Path to the log output directory (default: ./log_dir/)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='Seed for rng (default: 0)')

    # Parse
    print(DEVICE)
    args = parser.parse_args()
    print(args)

    run_environment(args)
