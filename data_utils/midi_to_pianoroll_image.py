import note_seq as ns
from note_seq.pianoroll_encoder_decoder import PianorollEncoderDecoder
from note_seq.melody_encoder_decoder import MelodyOneHotEncoding
from note_seq.sequences_lib import quantize_note_sequence
from PIL import Image
import numpy as np
from copy import deepcopy
from tqdm import tqdm

ped = PianorollEncoderDecoder(input_size=121)
QUANT = 4
NUM_SECS = 0
CROP = 0

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

def mid_file_to_input_target(midi_file):
    '''
    convert midi file to 2d numpy array
    Use this for inference or when you only need model input
    '''

    in_pr, tar_pr = mid_file_to_pr_input_target(midi_file, quant=QUANT, num_secs=NUM_SECS)
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

    pr = mid_file_to_pianoroll(midi_file, quant=QUANT)
    mask = np.zeros((ped.input_size, pr.num_steps))
    for step in pr.steps:
        mask[:, step] = ped.events_to_input(pr, step)
    return mask

def arr_to_image(arr, out_path):
    arr = arr # Note we crop the song to be 300 pixels in width
    Image.fromarray(255*arr).convert("RGB").save(out_path)

def image_to_class_indices(image_path):
    '''
    convert image to class indices used by PED to decode into pianorollseq
    '''
    cids = []
    img = np.asarray(Image.open(image_path).convert("RGB"))[:,:,0] # assume each channel is the same
    for i in range(img.shape[1]):
        pitches = np.where(img[:, i])[0]
        val=0
        for pitch in pitches:
            val += 1<<int(pitch) # Kinda Hacky, but works, python facilitates inf shift left, but numpy only goes to 63, need to convert to pure python here
        cids.append(val)
    return cids

def class_indices_to_pianoroll(cids):
    '''
    convert class indicies to pianorollseq
    '''
    events = []
    for cid in cids:
        events.append(ped.class_index_to_event(cid, []))
    #events = [ped.class_index_to_event(cid, []) for cid in cids]
    return ns.PianorollSequence(events_list=events, steps_per_quarter=QUANT,)

def image2midi(image_path, out_dir, mods=['converted']):
    '''
    convert an image back into a midi file
    use same name as initial file and place in converted folder
    '''
    ns.note_sequence_to_midi_file(class_indices_to_pianoroll(image_to_class_indices(image_path)).to_sequence(), f"{out_dir}/{mods[0]}/" + image_path.split("/")[-1].replace(".png",".mid"))

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
    '''

    import argparse
    import sys
    import glob
    import os

    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--midi_path", type=str, help="Folder Source of Midi files")
    parser.add_argument("-o", "--output_dir", type=str, help="Specify output dir for converted midi files, if converting image to midi, this is output midi path")
    parser.add_argument("-s", "--single_path", type=str, help="For debug and testing, run converter on single mid")
    parser.add_argument("-d", "--single_image", type=str, help="For debug and testing, run decoder on single image")
    parser.add_argument("-i", "--image_path", type=str, help="Folder Source for Image files")
    args = parser.parse_args()

    midi_path = args.midi_path
    image_path = args.image_path

    out = args.output_dir

    single_encode = args.single_path
    single_decode = args.single_image


    if single_encode is not None:
        inp, tar = mid_file_to_input_target(single_encode)
        arr_to_image(inp, '{out}_inp.png')
        arr_to_image(tar, '{out}_tar.png')
        exit(0)

    if single_decode is not None:
        ns.note_sequence_to_midi_file(class_indices_to_pianoroll(image_to_class_indices(single_decode)).to_sequence(), out)
        exit(0)

    if image_path is not None:
        os.makedirs(out, exist_ok=True) # Make the folder to hold the output files
        os.makedirs(out + '/converted', exist_ok=True)

        for image_file in tqdm(glob.glob(os.path.join(image_path,"*.png"))):
            try:
                image2midi(image_file, out_dir=out)
            except:
                print(f"{image_file} FAILED TO CONVERT!")

        exit(0)

    if midi_path is not None:
        os.makedirs(out, exist_ok=True) # Make the folder to hold the output files
        os.makedirs(out + '/input', exist_ok=True)
        os.makedirs(out + '/target', exist_ok=True)

        for midi_file in tqdm(glob.glob(os.path.join(midi_path,"*.mid*"))):
            try:
                midi2image(midi_file, out_dir=out)
            except:
                print(f"{midi_file} FAILED TO CONVERT!")
