# ============================================================================
# Training set-up for the minigrid environment
#
# Uses tensorboard for logging:
# https://pytorch.org/docs/stable/tensorboard.html
# ============================================================================

import argparse

import gym
from gym_minigrid import wrappers
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from flat_wrapper import MiniGridFlatWrapper
import dqn_flat_agent
import smi_flat_agent


# ========================================
# Helper methods for agent initialization
# ========================================

def init_agent(args, env, device='cpu'):

    if args.agent_type == 'dqn_flat':
        agent = dqn_flat_agent.DQNFlatAgent(
            num_actions=env.action_space.n,
            observation_shape=env.observation_space.shape,
            observation_dtype=torch.float32,  # NOTE hard-coded dtype
            history_size=args.history_size,
            gamma=args.discount_factor,
            min_replay_history=args.min_replay_history,
            update_period=args.update_period,
            target_update_period=args.target_update_frequency,
            epsilon_final=args.final_exploration,
            epsilon_decay_period=args.eps_decay_duration,
            memory_buffer_capacity=args.buffer_capacity,
            q_minibatch_size=args.q_batch_size,
            seed=args.seed,
            device=device,
        )
    elif args.agent_type == 'smirl_dqn_flat':
        agent = smi_flat_agent.SMiQFlatAgent(
            num_actions=env.action_space.n,
            observation_shape=env.observation_space.shape,
            observation_dtype=torch.float32,
            history_size=args.history_size,
            gamma=args.discount_factor,
            min_replay_history = args.min_replay_history,
            update_period=args.update_period,
            target_update_period=args.target_update_frequency,
            epsilon_final=args.final_exploration,
            epsilon_decay_period=args.eps_decay_duration,
            memory_buffer_capacity=args.buffer_capacity,
            q_minibatch_size=args.q_batch_size,
            vae_minibatch_size=args.vae_batch_size,
            seed=args.seed,
            device=device
        )
    else:
        agent = None


    """
    TODO delete, below for reference only
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
    """

    return agent


# ========================================
# Run environment
# ========================================

def run_environment(args: argparse.Namespace,
                    device: str = 'cpu',
                    logger: torch.utils.tensorboard.SummaryWriter = None):

    # ==
    # Set up environment
    env = gym.make(args.env_name)
    env = MiniGridFlatWrapper(env, use_tensor=False,
                              scale_observation=True,
                              scale_min=0, scale_max=10)

    # ==
    # Set up agent
    agent = init_agent(args, env, device=device)


    # ==
    # Start training
    print(f'Starting training, {args.num_episode} episodes')
    for episode_idx in range(args.num_episode):
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

            cumu_reward += reward
            timestep += 1

            if done:
                # Logging
                if args.log_dir is not None:
                    logger.add_scalar('Reward', cumu_reward,
                                      global_step=episode_idx)
                    if episode_idx % 10 == 0:
                        print(f'Epis {episode_idx}, Timesteps: {timestep}, Return: {cumu_reward}')

                else:
                    print(f'Epis {episode_idx}, Timesteps: {timestep}, Return: {cumu_reward}')

                # Agent logging TODO: not sure if this is the best practice
                agent.report(logger=logger, episode_idx=episode_idx)
                break

            # TODO: have some debugging print-out (e.g. every 100 episode) to make sure times and
            # things are good and training is happening

    env.close()
    if args.log_dir is not None:
        logger.close()


if __name__ == "__main__":
    # =====================================================
    # Setting up arguments
    parser = argparse.ArgumentParser(description='Training on MiniGrid environment')

    parser.add_argument('--num_episode', type=int, default=20, metavar='N',
                        help='number of episodes to run the environment for (default: 500)')
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-8x8-v0', metavar='N',
                        help='environment name (default: MiniGrid-Empty-8x8-v0)')

    # Agent parameters
    parser.add_argument('--agent_type', type=str, default='dqn_flat',
                        help='agent type to run: [dqn_flat, smirl_dqn_flat] '
                             '(default: dqn_flat)')
    parser.add_argument('--history_size', type=int, default=1, metavar='N',
                        help='number of most recent observations used to '
                             'construct a state (default: 1)')
    parser.add_argument('--buffer_capacity', type=int, default=20000, metavar='N',
                        help='replay buffer capacity (default: 20000)')
    parser.add_argument('--discount_factor', type=float, default=0.9, metavar='g',
                        help='gamma discount factor (default: 0.9)')

    parser.add_argument('--init_exploration', type=float, default=1.0, metavar='N',
                        help='initial e-greedy exploration value (default: 1.0)')
    parser.add_argument('--final_exploration', type=float, default=0.05, metavar='N',
                        help='final e-greedy exploration value (default: 0.05)')
    parser.add_argument('--eps_decay_duration', type=int, default=50000, metavar='N',
                        help='number of actions over which the initial '
                             'exploration rate is linearly annealed to the '
                             'final exploration rate (default: 50,000)')

    parser.add_argument('--min_replay_history', type=int, default=1024, metavar='N',
                        help='number of actions taken (transition stored) '
                             'before replay starts, also the number of actions'
                             'taken before the eps exploration begins to '
                             'decay (default: 1024)')
    parser.add_argument('--update_period', type=int, default=4, metavar='N',
                        help='num of actions selected between SGD updates '
                             '(default: 4)')
    parser.add_argument('--target_update_frequency', type=int, default=16, metavar='N',
                        help='frequency to update the target network, measured '
                             'in numbers of optimization steps taken (default: 16)')

    parser.add_argument('--q_batch_size', type=int, default=128, metavar='N',
                        help='minibatch size for training the policy (Q) '
                             'network (default: 128)')
    parser.add_argument('--vae_batch_size', type=int, default=256, metavar='N',
                        help='minibatch size for training the VAE density '
                             'estimation network (default: 256)')

    parser.add_argument('--log_dir', type=str, default=None, metavar='',
                        help='Path to the log output directory (default: None)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='Seed for rng (default: 0)')

    # Parse args
    args = parser.parse_args()
    print(args)

    # =====================================================
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # =====================================================
    # Initialize logging
    if args.log_dir is not None:
        # Tensorboard logger
        logger = SummaryWriter(log_dir=args.log_dir)
        # Add hyperparameters
        # TODO this will throw an exception for "None" hyperparmeters
        logger.add_hparams(hparam_dict=vars(args), metric_dict={})
    else:
        logger = None

    #
    run_environment(args, device=device, logger=logger)
