# OLD NAIVE METHOD TO CONVERT MIDIS TO IMAGES
# DO NOT USE AS IT TAKES TOO LONG, SAMPLING IS NAIVE (UNIFORMLY DISTRIBUTED)
from music21 import converter, instrument, note, chord
import json
import sys
import numpy as np
import os
from imageio import imwrite



def extractNote(element):
    return int(element.pitch.ps)

def extractDuration(element):
    return element.duration.quarterLength

def get_notes(notes_to_parse):

    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    durations = []
    notes = []
    start = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            if element.isRest:
                continue

            start.append(element.offset)
            notes.append(extractNote(element))
            durations.append(extractDuration(element))
                
        elif isinstance(element, chord.Chord):
            if element.isRest:
                continue
            for chord_note in element.notes:
                start.append(element.offset)
                durations.append(extractDuration(element))
                notes.append(extractNote(chord_note))

    return {"start":start, "pitch":notes, "dur":durations}

def midi2image(midi_path, prune=False, reps=1, out_dir=""):
    mid = converter.parse(midi_path)

    instruments = instrument.partitionByInstrument(mid)

    data = {}

    try:
        i=0
        for instrument_i in instruments.parts:
            notes_to_parse = instrument_i.recurse()

            if instrument_i.partName is None:
                data["instrument_{}".format(i)] = get_notes(notes_to_parse)
                i+=1
            else:
                data[instrument_i.partName] = get_notes(notes_to_parse)

    except:
        notes_to_parse = mid.flat.notes
        data["instrument_0".format(i)] = get_notes(notes_to_parse)

    resolution = 0.25

    for instrument_name, values in data.items():
        # https://en.wikipedia.org/wiki/Scientific_pitch_notation#Similar_systems
        upperBoundNote = 127
        lowerBoundNote = 21
        maxSongLength = 100

        index = 0
        prev_index = 0
        repetitions = 0
        while repetitions < reps: 
            if prev_index >= len(values["pitch"]):
                break

            matrix = np.zeros((upperBoundNote-lowerBoundNote,maxSongLength))

            pitchs = values["pitch"]
            durs = values["dur"]
            starts = values["start"]

            for i in range(prev_index,len(pitchs)):
                pitch = pitchs[i]

                dur = int(durs[i]/resolution)
                start = int(starts[i]/resolution)

                if dur+start - index*maxSongLength < maxSongLength:
                    for j in range(start,start+dur):
                        if j - index*maxSongLength >= 0:
                            temp = np.random.randint(0,9) if prune else True 
                            if temp:
                                matrix[pitch-lowerBoundNote,j - index*maxSongLength] = 255
                else:
                    prev_index = i
                    break
            out_file = os.path.join(out_dir, midi_path.split("/")[-1].replace(".mid",f"_{instrument_name}_{index}.png"))
            print(out_file)
            imwrite(out_file, matrix)
            index += 1
            repetitions+=1

if __name__=='__main__':
    '''
    example usage
    python convert_to_image.py -m train -o train_input -r 1 -p True
    '''
    
    import argparse 
    import sys
    import glob

    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--midi_path", type=str, help="Folder Source of Midi files")
    parser.add_argument("-p", "--prune", type=bool, help="Flag to prune nodes at random")
    parser.add_argument("-r", "--repetitions", type=int, help="Number or Repetitions")
    parser.add_argument("-o", "--output_dir", type=str, help="Specify output dir for converted midi files ")

    args = parser.parse_args()
    
    midi_path = args.midi_path 
    prune = args.prune
    reps = args.repetitions
    out_dir = args.output_dir

    os.makedirs(out_dir, exist_ok=True) # Make the folder to hold the output files
    
    for midi_file in glob.glob(os.path.join(midi_path,"*.mid")):
        midi2image(midi_file, prune=prune, reps=reps, out_dir=out_dir)
