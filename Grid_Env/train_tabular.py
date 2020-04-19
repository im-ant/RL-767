# ============================================================================
# Train the tabular agent
# ============================================================================

import argparse

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs import slippery_slope
from agents import tabular_SMiRL
from agents import tabular_modelbased


def _init_env_agent(args: argparse.Namespace):
    """
    Helper method for initializing the environment and agent
    """
    # ==
    # Initialize environment

    # Sample starting state coordinates
    coord_seed = args.env_seed
    if coord_seed is not None:
        coord_seed = coord_seed * 2
    coord_rng = np.random.RandomState(seed=coord_seed)
    init_ys = coord_rng.choice(args.grid_width, size=(args.num_init_states,),
                               replace=False)
    init_coords = [(y, 0) for y in init_ys]

    # Initialize environment
    env = slippery_slope.SlipperySlopeGridWorld(width=args.grid_width,
                                                lava_width=3,
                                                slip_prob=args.slip_prob,
                                                init_coords=init_coords,
                                                seed=args.env_seed)
    # ==
    # Initialize agent

    # State prior
    agent_state_prior = None
    if args.give_state_prior:
        agent_state_prior = None  # TODO change this

    # Transition matrix
    agent_trans_mat = None
    if args.give_transition_mat:
        agent_trans_mat = env.get_transition_matrix()

    n_states = (args.grid_width * args.grid_width)

    # Create agent
    if args.agent_type == 'smirl':
        agent = tabular_SMiRL.TabularMinSurpriseAgent(
            n_states=n_states,
            gamma=args.discount_factor,
            state_prior=agent_state_prior,
            trans_mat=agent_trans_mat,
            k_samples=args.rs_samples,
            trajectory_len=args.rs_length,
            seed=args.agent_seed
        )
    elif args.agent_type == 'model_based':
        agent = tabular_modelbased.TabularModelBasedAgent(
            n_states=n_states,
            gamma=args.discount_factor,
            state_prior=agent_state_prior,
            trans_mat=agent_trans_mat,
            k_samples=args.rs_samples,
            trajectory_len=args.rs_length,
            seed=args.agent_seed
        )
    else:
        agent = None
        assert agent is not None

    return env, agent


def get_transition_estimation_error(env, agent) -> float:
    """Compute the approximation error of the transition matrix"""

    # Not relevant if we've already given the agent a transition matrix
    if agent.trans_count is None:
        return -1.0

    # Environment transition
    env_T = env.get_transition_matrix()

    # Agent transition counts
    agent_T_counts = agent.trans_count
    # Normalize count
    t_rowsum = np.sum(agent_T_counts, axis=2, keepdims=True)
    agent_T = agent_T_counts / t_rowsum

    # Compare L2 distance
    l2_distance = np.linalg.norm((env_T.flatten() - agent_T.flatten()),
                                 ord=2)

    return l2_distance


def run_environment(args: argparse.Namespace,
                    logger: torch.utils.tensorboard.SummaryWriter = None):
    # ==
    # Initialize environment and agent
    env, agent = _init_env_agent(args)

    # ==
    # Save the transition matrix for later comparison
    env_trans = env.get_transition_matrix()

    # ==
    # Start training
    print(f'Start training for {args.num_episode} episodes')
    for episode_idx in range(args.num_episode):
        # Reset counter variables
        cumulative_reward = 0.0
        steps = 0

        # Reset environment and agent
        observation = env.reset()
        action = agent.begin_episode(observation)

        # ==
        # Run episode
        while True:
            # Interaction
            observation, reward, done, info = env.step(action)
            action = agent.step(observation, reward, done)

            # Counter variables
            cumulative_reward += reward
            steps += 1

            # TODO: need some way of recording the *recent* state occupancy
            # to evaluate the agent behaviour

            # ==
            # If done
            if done or steps >= args.max_steps:
                # ==
                # Compute error
                if episode_idx % 100 == 0:
                    t_err = get_transition_estimation_error(env, agent)

                # ==
                # Log
                if logger is None:
                    print(episode_idx, steps, cumulative_reward)
                else:
                    logger.add_scalar('Cumulative_reward', cumulative_reward,
                                      global_step=episode_idx)
                    logger.add_scalar('Steps', steps,
                                      global_step=episode_idx)
                    logger.add_scalar('Trans_l2_error', t_err,
                                      global_step=episode_idx)

                    if episode_idx % 100 == 0:
                        print(episode_idx, steps, cumulative_reward)

                # Agent self-report
                agent.report(logger=logger, episode_idx=episode_idx)

                break

    env.close()


if __name__ == "__main__":
    # ===
    # Arguments
    parser = argparse.ArgumentParser(description='Training on tabular environment')

    # Training set-up
    parser.add_argument('--num_episode', type=int, default=20, metavar='N',
                        help='number of episodes to run the environment for (default: 20)')
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-8x8-v0', metavar='N',
                        help='environment name (default: MiniGrid-Empty-8x8-v0)')  # not used

    # Environment set-up
    parser.add_argument('--grid_width', type=int, default=10, metavar='N',
                        help='width of the grid world (default: 10)')
    parser.add_argument('--slip_prob', type=float, default=0.3, metavar='P',
                        help='probability of slipping on the slope (default: 0.3)')
    parser.add_argument('--num_init_states', type=int, default=1, metavar='N',
                        help='number of possible starting states (default: 1)')
    parser.add_argument('--max_steps', type=int, default=20, metavar='N',
                        help='max steps per episode before termination (default: 20)')
    parser.add_argument('--env_seed', type=int, default=None, metavar='P',
                        help='environment seed (default None)')

    # Agent set-up
    parser.add_argument('--agent_type', type=str, default='smirl', metavar='R',
                        help='Pick agent type: [smirl, model_based]')
    parser.add_argument('--discount_factor', type=float, default=0.9, metavar='R',
                        help='gamma discount factor (default: 0.9)')
    parser.add_argument('--give_state_prior', type=bool, default=False,
                        help='give agent prior state distribution (default: False)')
    parser.add_argument('--give_transition_mat', type=bool, default=False,
                        help='give agent the perfect transition matrix (default: False)')
    parser.add_argument('--rs_samples', type=int, default=10, metavar='N',
                        help='number of trajectories to sample during random shooting'
                             '(default: 10)')
    parser.add_argument('--rs_length', type=int, default=10, metavar='N',
                        help='length of the trajectory to sample for random shooting'
                             '(default: 10)')
    parser.add_argument('--agent_seed', type=int, default=2, metavar='N',
                        help='agent seed (default: 2)')

    # Logging set-up
    parser.add_argument('--log_dir', type=str, default=None, metavar='',
                        help='Path to the log output directory (default: None)')

    # Parse arguments
    # print(DEVICE)
    args = parser.parse_args()
    print(args)

    # ==
    # Initialize the tensorboard logger
    if args.log_dir is not None:
        logger = SummaryWriter(log_dir=args.log_dir)

        # Save hyperparameters
        d_args = vars(args)
        args_dict = {k: (v if v is not None else 'None')
                     for k, v in d_args.items()}
        logger.add_hparams(hparam_dict=args_dict, metric_dict={})
    else:
        logger = None

    # ==
    # Run environment
    run_environment(args, logger=logger)

    if args.log_dir is not None:
        logger.close()
