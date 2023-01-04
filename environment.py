
import numpy as np
from gym import Env
from gym import spaces
import random




class CustomEnv(Env):

    def __init__(self, dataset):
        self.piano_length = 88
        self.look_forward = 3 # for state space
        self.dummy = 0
        self.reach = 13 # 8 white + 5 black
        # actions are to move hands or chord
        # todo: add more or make more simple
        self.action_to_fingering = {
            0: np.array([1, 5, 8, self.dummy, self.dummy]), # major
            1: np.array([1, 4, 8, self.dummy, self.dummy]), # minor
            2: np.array([1, 5, 9, self.dummy, self.dummy]), # augmented
            3: np.array([1, 4, 7, self.dummy, self.dummy]), # diminished
            4: np.array([1, 5, 8, 12, self.dummy]), # major 7
            5: np.array([1, 4, 8, 11, self.dummy]), # minor 7
            6: np.array([1, 5, 9, 12, self.dummy]), # augmented 7
            7: np.array([1, 4, 7, 10, self.dummy]), # diminished 7
            8: np.array([1, 5, 8, 11, self.dummy]), # dominant 7/ seventh
        }
        # numbers:
        self.num_hands = 2
        self.num_fingers = 10
        self.fingers_on_each = self.num_fingers // self.num_hands
        self.actions_per_hand = 2
        self.num_chords = len(self.action_to_fingering)

        # action space, root finger is at 0, but with chord then +1
        # todo add more restrictions on hand?
        self.action_space = spaces.MultiDiscrete([self.piano_length,
                                                  self.num_chords,
                                                  self.piano_length,
                                                  self.num_chords])
        self.observation_space = spaces.MultiDiscrete([self.piano_length+1]*self.num_fingers)
        # observation-space, plus 1 for dummy
        self.state = np.array([self.piano_length+1]*self.num_fingers)

        self.action_length = len(self.action_space)
        self.state_length = len(self.state)
        # each episode is a full song
        self.num_steps = 0
        self.episode_length = len(dataset)
        dataset.insert(0, '0')  # 0 as start value
        self.goals = dataset

    def check_contraints(self, actions, state):

        # fingers do not cross

        # hands do not cross

        # cannot go outside keyboard

        #

        return True


    def step(self, action):
        reward = 0
        # update state
        for idx in range(self.num_hands):
            # setting current state
            move_hand = action[self.actions_per_hand*idx]
            chord = action[self.actions_per_hand*idx+1]
            self.state[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] = move_hand * (self.action_to_fingering[chord] != 0)
            self.state[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] += self.action_to_fingering[chord]

            # any deviation is -1, non-deviation is +1 (thus smaller chords also desired)
            reward -= sum(1*(self.previous_action != action))
            #reward += sum(1*(self.previous_action == action))

        self.previous_action = action # update


        # Calculating the reward
        #if self.goals[self.num_steps].split('.') in self.state.astype(str):
        for note in self.goals[self.num_steps].split('.'):
            if note in self.state.astype(str):
                reward += 0 #10
            else:
                reward += -10

        # Checking if done
        self.num_steps += 1
        if self.num_steps == self.episode_length:
            done = True
        else:
            done = False

        # Setting the placeholder for info
        info = {}

        # Returning the step information
        return self.state, reward, done, info


    def render(self):
        # This is where you would write the visualization code
        # todo: add visuals of it playing
        return None

    def reset(self):
        # observation-space
        # initial
        #self.previous_action = np.array([30, 0, 50, 0])
        self.previous_action = self.action_space.sample()
        for idx in range(self.num_hands):
            # setting current state
            move_hand = self.previous_action[self.actions_per_hand*idx]
            chord = self.previous_action[self.actions_per_hand*idx+1]
            self.state[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] = move_hand * (self.action_to_fingering[chord] != 0)
            self.state[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] += self.action_to_fingering[chord]
        '''
        self.chord = np.array([0, 0])
        self.hand = np.array([30, 50])
        for idx, each_state in enumerate(self.state_types):
            self.state[each_state] = self.hand[idx]*(self.action_to_fingering[self.chord[idx]] != 0)
            self.state[each_state] += self.action_to_fingering[self.chord[idx]]
        '''
        #self.state = {
        #    "left hand": self.hand[0] + self.action_to_fingering[self.chord[0]],
        #    "right hand": self.hand[1] + self.action_to_fingering[self.chord[1]]
        #}
        self.num_steps = 0
        return self.state





