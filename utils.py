
import os
from music21 import *
import agents

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




