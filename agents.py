
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque


class random_walk(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, env, episodes):
        episodes = episodes  # 20 shower episodes
        scores = []
        for episode in range(1, episodes + 1):
            state = env.reset()
            done = False
            score = 0

            while not done:
                action = env.action_space.sample()
                n_state, reward, done, info = env.step(action)
                score += reward
            print('Episode:{} Score:{}'.format(episode, score))

            scores.append(score)
        return scores


class reinforce(torch.nn.Module):
    '''
    inspiration from:
    https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/12/REINFORCE-CartPole.html
    '''
    def __init__(self, state_length=4, action_shape=4, gamma=1.0, hidden_size=500):
        super().__init__()
        # for policy network
        self.policy = torch.nn.Sequential(
            nn.Linear(state_length, hidden_size),
            #nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, sum(action_shape)),
        )
        self.action_shape = tuple(action_shape)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=1e-5, amsgrad=True)
        self.gamma = gamma
        self.eps = np.finfo(np.float32).eps.item()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        output = self.policy(state)#.to(self.device)
        output_split = torch.split(output.flatten(), self.action_shape)
        actions = np.zeros(len(self.action_shape), dtype=int)
        log_prob = 0
        for idx, logits in enumerate(output_split):
            model = Categorical(logits=logits)
            action = model.sample()
            actions[idx] = action.item()
            log_prob += model.log_prob(action)

        return actions, log_prob

    def forward(self, env, episodes):

        scores_deque = deque(maxlen=100)
        scores = []
        for episode in range(episodes):
            saved_log_probs = []
            saved_states = np.zeros((env.episode_length, env.state_length))
            saved_actions = np.zeros((env.episode_length, env.action_length))
            rewards = []
            state = env.reset() # initial state is arbitrary
            # Collect trajectory
            done = False
            while not done: # environment always terminates
                # Sample the action from current policy
                action, log_prob = self.get_action(state)
                saved_log_probs.append(log_prob)
                # save
                saved_states[env.num_steps] = state
                saved_actions[env.num_steps] = action
                # reward and new state
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
            # Calculate total expected reward
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            # todo not working
            # Recalculate the total reward applying discounted factor
            discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
            G = sum([a * b for a, b in zip(discounts, rewards)])

            # Calculate the loss
            policy_loss = []
            for log_prob in saved_log_probs:
                # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
                policy_loss.append(-log_prob * G)
            # After that, we concatenate whole policy loss in 0th dimension
            policy_loss = sum(policy_loss)
            '''

            # similar to pytorch implementation
            R = 0
            policy_loss = []
            returns = []
            for r in rewards[::-1]: # going backward
                R = r + self.gamma * R
                returns.append(R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + self.eps) # trick
            #returns = torch.tensor(rewards) # seemed to work the best
            for log_prob, R in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
            policy_loss = sum(policy_loss)
            '''
            # Backpropagation
            if self.training:
                self.optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
                self.optimizer.step()

            print('Episode:{} Score:{}'.format(episode, sum(rewards)))
            #if e % self.print_every == 0:
            #    print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
            #if np.mean(scores_deque) >= 195.0:
            #    print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e - 100, np.mean(scores_deque)))
            #    break
        return scores, saved_states, saved_actions









class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def Qlearning():

    return 0







