
from environment import *
from agents import *
from utils import *

import matplotlib.pyplot as plt
import argparse
import torch
import datetime


def get_arguments(parser):
    parser.add_argument(
        "--mode",
        type=str,
        help="Defines whether what mode to run in.",
        default='stable_baselines3',
        choices=['train', 'test', 'traintest', 'stable_baselines3']
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name experiment.",
        default='testrun',
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default='reinforce_day03.01_time22.47',
        help="Name of model to be loaded.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for random behaviour.",
        default=42,
    )
    parser.add_argument(
        "--composer",
        default='debussy',
        help="Name of the author"
    )
    parser.add_argument(
        "--piece",
        default='deb_clai',
        help="Name of song/piece."
    )
    parser.add_argument(
        "--agent_type",
        help="The agent type to use when training.",
        default='reinforce',
        choices=['random_walk', 'reinforce']
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="How many episodes to run (in 'train'-mode).",
        default=20_000,
    )


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Runs associated methods for efficient piano fingering.")
    get_arguments(parser)
    args = parser.parse_args()
    # setting seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # defining simulation
    dataset = get_data(args.composer, args.piece)
    env = CustomEnv(dataset)

    if 'train' in args.mode:
        agent = get_agent(args.agent_type, len(env.observation_space), env.action_space.nvec)
        # train agent
        scores, _, _ = agent(env, args.episodes)
        now = datetime.datetime.now()
        date_time = now.strftime("date%d.%m_time%H.%M")
        save_name = args.experiment_name + '_' + args.agent_type + '_' + date_time
        torch.save(agent.state_dict(), f'saved/{save_name}.pth')
        # saving training plot
        plt.plot(scores)
        plt.ylabel('rewards')
        plt.xlabel('episodes')
        plt.savefig(f'results/{save_name}.jpg')
        #plt.show()

    if 'test' in args.mode:
        # loading agent
        agent = get_agent(args.agent_type, len(env.observation_space), env.action_space.nvec)
        if 'save_name' in globals():
            state_dict = torch.load(f"saved/{save_name}.pth")
        else:
            state_dict = torch.load(f"saved/{args.save_name}.pth")
        agent.load_state_dict(state_dict)
        agent.eval()
        # testing
        scores, states, actions = agent(env, 1)
        print('\n')

        notes = get_notes(env, actions)
        evaluate(actions, notes, env)


    elif 'stable_baselines3' in args.mode:
        from stable_baselines3 import PPO, A2C

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.episodes)

        vec_env = model.get_env()
        obs = vec_env.reset()
        states = np.zeros((env.episode_length, env.state_length))
        actions = np.zeros((env.episode_length, env.action_length))
        for i in range(env.episode_length):
            states[env.num_steps] = obs
            action, _states = model.predict(obs, deterministic=True)
            actions[env.num_steps] = action
            obs, reward, done, info = vec_env.step(action)
        env.close()
        notes = get_notes(env, actions)
        evaluate(actions, notes, env)