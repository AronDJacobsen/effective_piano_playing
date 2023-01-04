
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
        default='traintest',
        choices=['train', 'test', 'traintest']
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
        default=200,
    )


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Runs associated methods for Invariant Graph Neural Networks.")
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
        print(f'Score if all wrong')

        # analysis
        # precision
        correct = 0
        for step in range(env.episode_length):
            for note in env.goals[step].split('.'):
                if note in states[step].astype(int).astype(str):
                    correct += 1
        print(f'Number of correct: {correct} out of {env.episode_length}')
        print('\n')
        # unique
        print('Total unique states/grips: ', len(np.unique(states, axis=0)))
        differs = np.diff(actions, axis=0)**2 # for later
        for idx, hand in enumerate(['left hand', 'right hand']):
            print(f'Unique {hand} placements: ',
                  np.unique(actions[:, idx*env.num_hands]).astype(int).tolist())
            print(f'Unique {hand} chords: ',
                  np.unique(actions[:, idx*env.num_hands+1]).astype(int).tolist())
        print('\n')
        # how many changes
        differs = np.diff(actions, axis=0)**2
        print('Total changes:', sum(differs.sum(axis=1) != 0))
        differs.sum(axis=0)
        for idx, hand in enumerate(['left hand', 'right hand']):
            diff = sum(differs[:, idx * env.num_hands:idx * env.num_hands+env.num_hands].sum(axis=1) != 0)
            print(f'{hand} changes: ', diff)

        print('\nEvaluation finished')


