
import numpy as np
from gym import Env
from gym import spaces
from embeddings import GloveEmbedding



class CustomEnv(Env):

    def __init__(self, dataset):
        self.piano_length = 88
        self.look_forward = 2 # for state space
        self.dummy = 0
        self.reach = 13 # 8 white + 5 black
        # actions are to move hands or chord
        # todo: add more or make more simple
        self.action_to_fingering = {
            0: np.array([1, 5, 8, self.dummy, self.dummy]), # major
            1: np.array([1, 4, 8, self.dummy, self.dummy]), # minor
            2: np.array([1, 5, 9, self.dummy, self.dummy]), # augmented
            3: np.array([1, 4, 7, self.dummy, self.dummy]), # diminished
            #4: np.array([1, 5, 8, 12, self.dummy]), # major 7
            #5: np.array([1, 4, 8, 11, self.dummy]), # minor 7
            #6: np.array([1, 5, 9, 12, self.dummy]), # augmented 7
            #7: np.array([1, 4, 7, 10, self.dummy]), # diminished 7
            #8: np.array([1, 5, 8, 11, self.dummy]), # dominant 7/ seventh
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
        # observation-space, plus 1 for dummy, and plus 1 for current state
        self.observation_space = spaces.MultiDiscrete([self.piano_length+1]*(self.look_forward+1))
        self.state = np.array([self.piano_length+1]*(self.look_forward+1))
        self.action_length = len(self.action_space)
        self.state_length = len(self.state)
        # each episode is a full song
        self.num_steps = 0
        self.episode_length = len(dataset)
        self.initial_goals = np.array(dataset)
        # modifying dataset to avoid errors in loop
        #dataset.insert(0, '0')  # 0 as start value
        dataset.extend([0] * self.look_forward) # to avoid index error
        self.goals = np.array(dataset)
        self.observations = np.zeros(len(self.goals))
        for idx, goal in enumerate(self.goals):
            self.observations[idx] = np.array(self.goals[idx].split('.')).astype(int).mean()


    def check_contraints(self, actions, state):

        # fingers do not cross

        # hands do not cross

        # cannot go outside keyboard

        #

        return True


    def step(self, action):

        # initialize
        reward = 0

        # update state
        self.state = self.observations[self.num_steps:self.num_steps + self.look_forward + 1]

        '''
        for idx in range(self.num_hands):
            # setting current state
            move_hand = action[self.actions_per_hand*idx]
            chord = action[self.actions_per_hand*idx+1]
            self.state[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] = move_hand * (self.action_to_fingering[chord] != 0)
            self.state[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] += self.action_to_fingering[chord]
        '''
        # any deviation is -1, non-deviation is +1 (thus smaller chords also desired)
        #reward -= sum(1*(self.previous_action != action))
        #reward += sum(1*(self.previous_action == action))
        # todo: add punishment for crossing hands?
        self.previous_action = action # update

        # finding current notes
        notes = np.zeros(self.num_fingers)
        for idx in range(self.num_hands):
            move_hand = action[self.actions_per_hand*idx]
            chord = action[self.actions_per_hand*idx+1]
            notes[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] = move_hand * (self.action_to_fingering[chord] != 0)
            notes[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] += self.action_to_fingering[chord]

        notes = notes.astype(int).astype(str)
        # Calculating the reward
        for goal_note in self.goals[self.num_steps].split('.'):
            if goal_note in notes:
                reward += 10
            else:
                reward -= 0#10

        # Checking if done
        # plus 1 since starts at 0
        if self.num_steps+1 == self.episode_length:
            done = True
        else:
            done = False
            # updating
            self.num_steps += 1

        # Setting the placeholder for info
        info = {}

        # Returning the step information
        return self.state, reward, done, info


    def render(self):
        # This is where you would write the visualization code
        # todo: add visuals of it playing
        return None

    def reset(self):
        # starting steps in song
        self.num_steps = 0
        # observation-space
        # initial
        #self.previous_action = np.array([30, 0, 50, 0])
        self.previous_action = self.action_space.sample() # necessary to sample?
        self.state = self.observations[self.num_steps:self.num_steps+self.look_forward+1]
        '''
        for idx in range(self.num_hands):
            # setting current state
            move_hand = self.previous_action[self.actions_per_hand*idx]
            chord = self.previous_action[self.actions_per_hand*idx+1]
            self.state[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] = move_hand * (self.action_to_fingering[chord] != 0)
            self.state[idx*self.fingers_on_each:(idx+1)*self.fingers_on_each] += self.action_to_fingering[chord]
        '''

        return self.state





