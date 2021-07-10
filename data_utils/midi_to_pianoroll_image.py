import note_seq as ns
from note_seq.pianoroll_encoder_decoder import PianorollEncoderDecoder
from note_seq.melody_encoder_decoder import MelodyOneHotEncoding
from note_seq.sequences_lib import quantize_note_sequence
from PIL import Image
import numpy as np
from copy import deepcopy

ped = PianorollEncoderDecoder(input_size=121)
OneHotEncoderDecoder = MelodyOneHotEncoding(0, 128)
num_classes = OneHotEncoderDecoder.num_classes

def mid_file_to_melody(mid_file_path, quant=1, num_secs=10):
    '''
    converts a file path to a ns.Melody() object, need to specify the number of counts given to a quarter note for note_sequence quantization

    Args:
        - mid_file_path - str, path to the mid file, can be absolute or relative
        - quant - number of steps a quarter note represents, this is required for quantization, bigger number means more data but better note accuracy
        - num_secs - the max number of seconds of music we are interested in listening to, if set to 0, we will not truncate
    '''
    f_ns = ns.midi_file_to_note_sequence(mid_file_path)

    main_instrument = ns.infer_melody_for_sequence(f_ns)
    notes = [note for note in f_ns.notes if note.instrument == main_instrument]

    del f_ns.notes[:]
    f_ns.notes.extend(
        sorted(notes, key=lambda note: note.start_time))

    melon = ns.Melody()
    melon.from_quantized_sequence(quantize_note_sequence(f_ns, quant), instrument=main_instrument, ignore_polyphonic_notes=True)

    if num_secs:
        if melon.total_time > num_secs:
            melon = ns.extract_subsequence(
                melon, 0, num_secs)

    return melon

def mid_file_to_pr_input_target(mid_file_path, quant=4, num_secs=0):
    '''
    converts a file path to a ns.Melody() object, need to specify the number of counts given to a quarter note for note_sequence quantization

    Args:
        - mid_file_path - str, path to the mid file, can be absolute or relative
        - quant - number of steps a quarter note represents, this is required for quantization, bigger number means more data but better note accuracy
        - num_secs - the max number of seconds of music we are interested in listening to, if set to 0, we will not truncate
    '''
    f_ns = ns.midi_file_to_note_sequence(mid_file_path)
    if num_secs != 0:
        if f_ns.total_time > num_secs:
            f_ns = ns.extract_subsequence(
                f_ns, 0, num_secs)
    pr_target = ns.PianorollSequence(quantized_sequence=quantize_note_sequence(f_ns, quant))

    s_ns = deepcopy(f_ns)
    main_instrument = ns.infer_melody_for_sequence(s_ns)
    notes = [note for note in s_ns.notes if note.instrument == main_instrument]
    del s_ns.notes[:]
    s_ns.notes.extend(
        sorted(notes, key=lambda note: note.start_time))

    pr_input = ns.PianorollSequence(quantized_sequence=quantize_note_sequence(s_ns, quant))

    return pr_input, pr_target

def mid_file_to_pianoroll(mid_file_path, quant=4, num_secs=0, top_only=True):
    '''
    converts a file path to a ns.Melody() object, need to specify the number of counts given to a quarter note for note_sequence quantization

    Args:
        - mid_file_path - str, path to the mid file, can be absolute or relative
        - quant - number of steps a quarter note represents, this is required for quantization, bigger number means more data but better note accuracy
        - num_secs - the max number of seconds of music we are interested in listening to, if set to 0, we will not truncate
        - top_only - extract the main voice only to pianoroll
    '''

    f_ns = ns.midi_file_to_note_sequence(mid_file_path)

    if num_secs != 0:
        if f_ns.total_time > num_secs:
            f_ns = ns.extract_subsequence(
                f_ns, 0, num_secs)
    if top_only:
        main_instrument = ns.infer_melody_for_sequence(f_ns)
        notes = [note for note in f_ns.notes if note.instrument == main_instrument]

        del f_ns.notes[:]
        f_ns.notes.extend(
            sorted(notes, key=lambda note: note.start_time))

    proll = ns.PianorollSequence(quantized_sequence=quantize_note_sequence(f_ns, quant))

    return proll

def convert_to_onehot(ns_melody, seq_len=0):
    '''
    convert note_seq.Melody() object to onehot numpy array of length seq_len

    Args:
        ns_melody - note_seq.Melody() object created from midi notes
        seq_len - number of columns for desired sequence, if zero, just use encoded melody length
    Returns:
        np.array - dimension of 130 x seq_len
    '''

    onehot_mel = [ OneHotEncoderDecoder.encode_event(hot) for hot in ns_melody]

    if seq_len == 0:
        seq_len = len(onehot_mel)

    out = np.zeros((num_classes, seq_len))

    len(ns_melody)
    out[np.arange(onehot_mel.size), onehot_mel] = 1

    return out

def mid_to_png(mid_file_path, outfile, quant=4, rgb=False):
    '''
    convert a midi file to an png image

    args:
        - mid_file_path: str, path to the desired mid file to read
        - outfile: str, name of the desired output image
        - quant: note quantization
        - rgb: add more channels with repeated melody
    '''

    convert_to_onehot(mid_file_to_melody(mid_file_path, quant=quant))

midi_file = 'MuseData/test/bach.cant.0003_midip_02.mid'

#f_ns = ns.midi_file_to_note_sequence(midi_file)
#mel = ns.infer_melody_for_sequence(f_ns)
#notes = [note for note in f_ns.notes if note.instrument == mel]
#
#del f_ns.notes[:]
#f_ns.notes.extend(
#    sorted(notes, key=lambda note: note.start_time))
#melon = ns.Melody()
#
#q_ns = quantize_note_sequence(f_ns, 1)
#
#print(q_ns)
#melon.from_quantized_sequence(q_ns, instrument=mel, ignore_polyphonic_notes=True)
#
#for ev in melon:
#    print(ev)

#melon = mid_file_to_melody(midi_file)
#
#onehot_mel = [ OneHotEncoderDecoder.encode_event(hot) for hot in melon]
#
#print(onehot_mel)
#convert_back = [ OneHotEncoderDecoder.decode_event(hot) for hot in onehot_mel]
#
#print(convert_back)
def mid_file_to_input_target(midi_file):
    '''
    convert midi file to 2d numpy array
    Use this for inference or when you only need model input
    '''

    in_pr, tar_pr = mid_file_to_pr_input_target(midi_file)
    input_mask = np.zeros((ped.input_size, in_pr.num_steps))
    target_mask = np.zeros((ped.input_size, tar_pr.num_steps))
    for step in in_pr.steps:
        input_mask[:, step] = ped.events_to_input(in_pr, step)

    for step in tar_pr.steps:
       target_mask[:, step] = ped.events_to_input(tar_pr, step)

    return input_mask, target_mask

def mid_file_to_arr(midi_file):
    '''
    convert midi file to 2d numpy array
    Use this for inference or when you only need model input
    '''

    pr = mid_file_to_pianoroll(midi_file)
    mask = np.zeros((ped.input_size, pr.num_steps))
    for step in pr.steps:
        mask[:, step] = ped.events_to_input(pr, step)
    return mask

def mid_file_to_image(midi_file):
    '''
    convert midi file to an image using the piano roll format
    '''

    # sample = mask[:,:,None]*np.ones(3,dtype=int)[None,None,:]
    # sample = 255*sample.astype(int)
    im = Image.fromarray(255*mid_file_to_arr(midi_file)).convert("RGB")
    im.save("swagger.jpg")

def arr_to_image(arr, out_path):
    Image.fromarray(255*arr).convert("RGB").save(out_path)

def midi2image(midi_path, out_dir, mods=['input', 'target']):
    '''
    create input and target masks as images

    input and taget folders will be created in the output dir
    '''

    inp, tar =  mid_file_to_input_target(midi_file)
    arr_to_image(inp, f"{out_dir}/{mods[0]}/" + midi_path.split("/")[-1].replace(".mid", ".png")) # might need to change to os.path.join
    arr_to_image(tar, f"{out_dir}/{mods[1]}/" + midi_path.split("/")[-1].replace(".mid", ".png"))

if __name__ == "__main__":
    '''
    example usage
    python midi_to_pianoroll_image.py -m train -o train_input
    TODO:
     - Add parameter for number of seconds of midi to keep
     - Add parameter for quantization of midi
    '''

    import argparse
    import sys
    import glob
    import os

    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--midi_path", type=str, help="Folder Source of Midi files")
    parser.add_argument("-o", "--output_dir", type=str, help="Specify output dir for converted midi files ")
    parser.add_argument("-s", "--single_path", type=str, help="For debug and testing, run converter on single image")
    args = parser.parse_args()

    midi_path = args.midi_path
    out_dir = args.output_dir
    single = args.single_path
    
    if single is not None:
        inp, tar = mid_file_to_input_target(single)
        arr_to_image(inp, 'debug_inp.png')
        arr_to_image(tar, 'debug_tar.png')
        exit(0)
    os.makedirs(out_dir, exist_ok=True) # Make the folder to hold the output files
    os.makedirs(out_dir + '/input', exist_ok=True)
    os.makedirs(out_dir + '/target', exist_ok=True)

    for midi_file in glob.glob(os.path.join(midi_path,"*.mid")):
        print(midi_file)
        try:
            midi2image(midi_file, out_dir=out_dir)
        except:
            print(f"{midi_file} FAILED TO CONVERT!")
