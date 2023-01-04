
import os
from music21 import *
import agents
import numpy as np

def train(env, episodes):

    episodes = episodes  # 20 shower episodes
    for episode in range(1, episodes +1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))


def get_data(composer, piece):
    # Loading the list of chopin's midi files as stream
    filepath = os.getcwd() + f"/data/{composer}/"
    tr = filepath + f"{piece}.mid" # todo some assert?
    midi = converter.parse(tr)
    notes = [] # contains midi numbering (e.g. A0 starts at 21, to be adjust to value 1)
    song = instrument.partitionByInstrument(midi)
    for part in song.parts:
        pick = part.recurse()
        for element in pick:
            # nameWithOctave
            # MIDI number
            if isinstance(element, note.Note):
                notes.append(str(element.pitch.midi-20))
            elif isinstance(element, chord.Chord):
                #notes[idx].append(".".join(str(n) for n in element.normalOrder))
                #notes.append(".".join(str(note.Note(idx).pitch.midi) for idx in element.normalOrder))
                notes.append(".".join(str(pitch.midi-20) for pitch in element.pitches))
    return notes

def get_agent(agent_type, state_length, action_shape):

    if agent_type == 'random_walk':
        return agents.random_walk()
    elif agent_type == 'reinforce':
        return agents.reinforce(state_length=state_length, action_shape=action_shape)


def get_notes(env, actions):
    # actions to notes
    notes = np.zeros((env.episode_length, env.num_fingers))
    for a_idx, action in enumerate(actions):
        for idx in range(env.num_hands):
            move_hand = action[env.actions_per_hand * idx]
            chord = action[env.actions_per_hand * idx + 1]
            notes[a_idx, idx * env.fingers_on_each:(idx + 1) * env.fingers_on_each] = move_hand * (
                        env.action_to_fingering[chord] != 0)
            notes[a_idx, idx * env.fingers_on_each:(idx + 1) * env.fingers_on_each] += env.action_to_fingering[chord]
    notes = notes.astype(int).astype(str)
    return notes



def evaluate(actions, notes, env):
    print('\n')
    # analysis
    # precision
    correct = 0
    total = 0
    for step in range(env.episode_length):
        for note in env.initial_goals[step].split('.'):
            if note in notes[step]:
                correct += 1
            total += 1
    print(f'Number of correct: {correct} out of {total}')
    print('\n')
    # unique
    print('Total unique actions: ', len(np.unique(actions, axis=0)))
    differs = np.diff(actions, axis=0) ** 2  # for later
    for idx, hand in enumerate(['left hand', 'right hand']):
        print(f'Unique {hand} placements: ',
              np.unique(actions[:, idx * env.num_hands]).astype(int).tolist())
        print(f'Unique {hand} chords: ',
              np.unique(actions[:, idx * env.num_hands + 1]).astype(int).tolist())
    print('\n')
    # how many changes
    differs = np.diff(actions, axis=0) ** 2
    print('Total changes:', sum(differs.sum(axis=1) != 0))
    differs.sum(axis=0)
    for idx, hand in enumerate(['left hand', 'right hand']):
        diff = sum(differs[:, idx * env.num_hands:idx * env.num_hands + env.num_hands].sum(axis=1) != 0)
        print(f'{hand} changes: ', diff)

    print('\nEvaluation finished')





