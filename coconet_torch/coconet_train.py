import os
import time

import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='Run coconet train session with hyperparams as flags')
parser.add_argument('--dataset', default=None, choices=['Jsb16thSeparated', 'MuseData', 'Nottingham', 'PianoMidiDe'])
parser.add_argument('--quantization_level', default=0.125, type=float,help='Path to the directory where checkpoints and '
                    'summary events will be saved during training and '
                    'evaluation. Multiple runs can be stored within the '
                    'parent directory of `logdir`. Point TensorBoard '
                    'to the parent directory of `logdir` to see all '
                    'your runs.')

def _hyperparams_from_flags():
    """Instantiate hyperparameters based on flags specified in CLI call."""

    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    print(_hyperparams_from_flags())
