import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import collections

import lib_hparams

class CoconetGraph(nn.Module):
    '''Model for predicting autofills given context.'''

    def __init__(self, is_training, hparams, placeholders=None, direct_inputs=None, use_placeholders=True):
        super(CoconetGraph, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.num_pitches = hparams.num_pitches
        self.num_instruments = hparams.num_instruments
        self.is_training = is_training
        self.placeholders = placeholders
        self._direct_inputs = direct_inputs
        self._use_placeholders = use_placeholders
        self.hiddens = []
        self.popstats_by_batchstat = collections.OrderedDict()
        self.build()

    @property
    def use_placeholders(self):
        return self._use_placeholders

    @use_placeholders.setter
    def use_placeholders(self, use_placeholders):
        self._use_placeholders = use_placeholders

    @property
    def inputs(self):
        if self.use_placeholders:
            return self.placeholders
        else:
            return self.direct_inputs

    @property
    def direct_inputs(self):
        return self._direct_inputs

    @direct_inputs.setter
    def direct_inputs(self, direct_inputs):
        if set(direct_inputs.keys()) != set(self.placeholders.keys()):
            raise AttributeError('Need to have pianorolls, masks, lengths.')
        self._direct_inputs = direct_inputs

    @property
    def pianorolls(self):
        return self.inputs['pianorolls']

    @property
    def masks(self):
        return self.inputs['masks']

    @property
    def lengths(self):
        return self.inputs['lengths']

    def build(self):
        '''Builds the graph.'''
        featuremaps = self.get_convnet_input()
        self.residual_init()

        conv_arch = self.hparams.get_convnet_arch()
        layers = conv_arch.layers

        n = len(layers)
        torch_layers = []
        for i, layer in enumerate(layers):
            print(layer)

    def get_convnet_input(self):
        """Returns concatenates masked out pianorolls with their masks."""
        # pianorolls, masks = self.inputs['pianorolls'], self.inputs[
        #     'masks']
        pianorolls, masks = self.pianorolls, self.masks
        pianorolls *= 1. - masks
        if self.hparams.mask_indicates_context:
            # flip meaning of mask for convnet purposes: after flipping, mask is hot
            # where values are known. this makes more sense in light of padding done
            # by convolution operations: the padded area will have zero mask,
            # indicating no information to rely on.
            masks = 1. - masks
        return torch.cat([pianorolls, masks], dim=3)

    def residual_init(self):
        if not self.hparams.use_residual:
            return
        self.residual_period = 2
        self.output_for_residual = None
        self.residual_counter = -1

    def residual_reset(self):
        self.output_for_residual = None
        self.residual_counter = 0

    def residual_save(self, x):
        if not self.hparams.use_residual:
            return
        if self.residual_counter % self.residual_period == 1:
            self.output_for_residual = x
                
def get_placeholders(hparams):
    return dict(
        pianorolls=torch.rand(
            [2, hparams.batch_size, hparams.num_pitches, hparams.num_instruments]),
        masks=torch.rand(
            [2, hparams.batch_size, hparams.num_pitches, hparams.num_instruments]),
        lengths = torch.rand([64]))

def build_graph(is_training,
                hparams, 
                placeholders=None,
                direct_inputs=None,
                use_palceholders=True):
    '''Builds the model graph.'''
    if placeholders is None and use_palceholders:
        placeholders = get_placeholders(hparams)

    model = CoconetGraph(is_training=is_training, 
        hparams=hparams,
        placeholders=placeholders,
        direct_inputs=direct_inputs,
        use_placeholders=use_palceholders)
    return model

if __name__ == '__main__':
    hparams = lib_hparams.Hyperparameters(**{'max_pitch': 81, 'min_pitch': 36, 'num_instruments': 4})
    graph = build_graph(is_training=True, hparams=hparams)