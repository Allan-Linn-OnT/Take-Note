from collections import namedtuple
import os
import time
import shutil

import numpy as np
import torch
import argparse

from torch._C import memory_format

import lib_data
import lib_graph
import lib_hparams
import lib_util

import six
from six.moves import range
from six.moves import zip

parser = argparse.ArgumentParser(
    description='Run coconet train session with hyperparams as flags')
parser.add_argument_group('Train Params')
parser.add_argument('--data_dir', default=None)
parser.add_argument('--logdir', default=None)
parser.add_argument('--log_progress', default=True)

parser.add_argument_group('Dataset')
parser.add_argument('--dataset', default='Jsb16thSeparated',
                    choices=['Jsb16thSeparated', 'MuseData', 'Nottingham', 'PianoMidiDe'])
parser.add_argument('--quantization_level', default=0.125, type=float,
                    help='Quantization duration. For qpm=120, notated quarter note equals 0.5.')
parser.add_argument('--num_instruments', default=4,
                    help='Maximum number of instruments that appear in this dataset. Use 0 if not separating instruments and hence does not matter how many there are.')
parser.add_argument('--separate_instruments', default=True,
                    help='Separate instruments into different input feature maps or not.')
parser.add_argument('--crop_piece_len', default=64,
                    help='The number of time steps included in a crop')

parser.add_argument_group('Model Architecture')
parser.add_argument('--architecture', default='straight',
                    help='Convnet style. Choices: straight')
parser.add_argument('--use_sep_conv', default=False,
                    help='Use depthwise separable convolutions.')
parser.add_argument('--sep_conv_depth_multiplier', default=1,
                    help='Depth multiplier for depthwise separable convolutions')
parser.add_argument('--num_initial_regular_conv_layers', default=2,
                    help='The number of regular convolutional layers to start with when using depthwise separable convolutional layers.')
parser.add_argument('--num_pointwise_splits', default=1,
                    help='Num of splits on the pointwise convolution stage in depthwise separable convolutions.')
parser.add_argument('--interleave_split_every_n_layers', default=1,
                    help='Num of split pointwise layers to interleave between full pointwise layers.')
parser.add_argument('--num_dilation_blocks', default=3,
                    help='The number of dilation blocks that starts from dilation rate=1')
parser.add_argument('--dilate_time_only', default=False,
                    help='If set, only dilates the time dimension and not pitch')
parser.add_argument('--repeat_last_dilation_level', default=False,
                    help='If set, repeats the last dilation rate')
# Higher level arch params
parser.add_argument('--num_layers', default=16,
                    help='The number of convolutional layers for architecture that do not use dilated convs')
parser.add_argument('--num_filters', default=128,
                    help='The number of filters for each convolutional layer.')
parser.add_argument('--use_residual', default=True,
                    help='Add residual connections or not.')
parser.add_argument('--batch_size', default=20,
                    help='The batch size for training and validating the model.')

parser.add_argument_group('Mask Related')
parser.add_argument('--maskout_method', default='bernoulli',
                    help='Choices: bernoulli or orderless')
parser.add_argument('--mask_indicates_context', default=True,
                    help='Feed inverted mask into convnet so that zero-padding makes sense.')
parser.add_argument('--optimize_mask_only', default=False,
                    help='Optimize masked predictions only.')
parser.add_argument('--rescale_loss', default=True,
                    help='Rescale loss based on context size.')
parser.add_argument('--patience', default=5,
                    help='Number of epochs to wait for improvement before decaying learning rate.')
parser.add_argument('--corrupt_ratio', default=0.5,
                    help='Fraction of variables to mask out.')
parser.add_argument('--num_epochs', default=0,
                    help='The number of epochs to train the model. Default is 0 which means to run until terminated manually.')
parser.add_argument('--save_model_secs', default=360,
                    help='The number of seconds between saving each checkpoint.')
parser.add_argument('--eval_freq', default=5,
                    help='The number of training iterations before validation.')
parser.add_argument('--run_id', default='',
                    help='A run_id to add to directory names to avoid accidentally overwriting when testing same setups.')


def _hyperparams_from_flags():
    """Instantiate hyperparameters based on flags specified in CLI call."""

    args = vars(parser.parse_args())
    args = namedtuple('ObjectName', args.keys())(*args.values())
    return args


def estimate_popstats(unused_sv, sess, m, dataset, unused_hparams):
    """Averages over mini batches for population statistics for batch norm."""
    print('Estimating population statistics...')
    tfbatchstats, tfpopstats = list(
        zip(*list(m.popstats_by_batchstat.items())))

    nepochs = 3
    nppopstats = [lib_util.AggregateMean('') for _ in tfpopstats]
    for _ in range(nepochs):
        batches = (
            dataset.get_featuremaps().batches(size=m.batch_size, shuffle=True))
        for unused_step, batch in enumerate(batches):
            feed_dict = batch.get_feed_dict(m.placeholders)
            print(feed_dict)
            npbatchstats = sess.run(tfbatchstats, feed_dict=feed_dict)
            for nppopstat, npbatchstat in zip(nppopstats, npbatchstats):
                nppopstat.add(npbatchstat)
    nppopstats = [nppopstat.mean for nppopstat in nppopstats]

    _print_popstat_info(tfpopstats, nppopstats)

    # Update tfpopstat variables.
    for unused_j, (tfpopstat, nppopstat) in enumerate(
            zip(tfpopstats, nppopstats)):
        tfpopstat.load(nppopstat)


def run_epoch(supervisor, sess, m, dataset, hparams, eval_op, experiment_type,
              epoch_count):
    """Runs an epoch of training or evaluate the model on given data."""
    # reduce variance in validation loss by fixing the seed
    data_seed = 123 if experiment_type == 'valid' else None
    with lib_util.numpy_seed(data_seed):
        batches = (
            dataset.get_featuremaps().batches(
                size=m.batch_size, shuffle=True, shuffle_rng=data_seed))

    losses = lib_util.AggregateMean('losses')
    losses_total = lib_util.AggregateMean('losses_total')
    losses_mask = lib_util.AggregateMean('losses_mask')
    losses_unmask = lib_util.AggregateMean('losses_unmask')

    start_time = time.time()
    for unused_step, batch in enumerate(batches):
        # Evaluate the graph and run back propagation.
        fetches = [
            m.loss, m.loss_total, m.loss_mask, m.loss_unmask, m.reduced_mask_size,
            m.reduced_unmask_size, m.learning_rate, eval_op
        ]
        feed_dict = batch.get_feed_dict(m.placeholders)
        print(feed_dict)
        (loss, loss_total, loss_mask, loss_unmask, reduced_mask_size,
         reduced_unmask_size, learning_rate, _) = sess.run(
             fetches, feed_dict=feed_dict)

        # Aggregate performances.
        losses_total.add(loss_total, 1)
        # Multiply the mean loss_mask by reduced_mask_size for aggregation as the
        # mask size could be different for every batch.
        losses_mask.add(loss_mask * reduced_mask_size, reduced_mask_size)
        losses_unmask.add(loss_unmask * reduced_unmask_size,
                          reduced_unmask_size)

        if hparams.optimize_mask_only:
            losses.add(loss * reduced_mask_size, reduced_mask_size)
        else:
            losses.add(loss, 1)

    # Collect run statistics.
    run_stats = dict()
    run_stats['loss_mask'] = losses_mask.mean
    run_stats['loss_unmask'] = losses_unmask.mean
    run_stats['loss_total'] = losses_total.mean
    run_stats['loss'] = losses.mean
    if experiment_type == 'train':
        run_stats['learning_rate'] = float(learning_rate)

    # Make summaries.
#   if FLAGS.log_progress:
#     summaries = tf.Summary()
#     for stat_name, stat in six.iteritems(run_stats):
#       value = summaries.value.add()
#       value.tag = '%s_%s' % (stat_name, experiment_type)
#       value.simple_value = stat
#     supervisor.summary_computed(sess, summaries, epoch_count)

#   tf.logging.info(
#       '%s, epoch %d: loss (mask): %.4f, loss (unmask): %.4f, '
#       'loss (total): %.4f, log lr: %.4f, time taken: %.4f',
#       experiment_type, epoch_count, run_stats['loss_mask'],
#       run_stats['loss_unmask'], run_stats['loss_total'],
#       np.log(run_stats['learning_rate']) if 'learning_rate' in run_stats else 0,
#       time.time() - start_time)

    return run_stats['loss']


def main(unused_argv, FLAGS):
    """Builds the graph and then runs training and validation."""

    assert FLAGS.data_dir is not None, 'No Input directory was provided.'

    print(FLAGS.maskout_method, 'separate', FLAGS.separate_instruments)

    hparams = _hparams_from_flags()

    # Get data.
    print('dataset:', FLAGS.dataset, FLAGS.data_dir)
    print('current dir:', os.path.curdir)
    train_data = lib_data.get_dataset(FLAGS.data_dir, hparams, 'train')
    valid_data = lib_data.get_dataset(FLAGS.data_dir, hparams, 'valid')
    print('# of train_data:', train_data.num_examples)
    print('# of valid_data:', valid_data.num_examples)

    if train_data.num_examples < hparams.batch_size:
        print('reducing batch_size to %i' % train_data.num_examples)
        hparams.batch_size = train_data.num_examples

    train_data.update_hparams(hparams)

    # Save hparam configs.
    logdir = os.path.join(FLAGS.logdir, hparams.log_subdir_str)
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.mkdir(logdir)
    config_fpath = os.path.join(logdir, 'config')
    print('Writing to %s', config_fpath)
    with open(config_fpath, 'w') as p:
        hparams.dump(p)

    # Build the graph and subsequently running it for train and validation.
    # Build placeholders and training graph, and validation graph with reuse.
    m = lib_graph.build_graph(is_training=True, hparams=hparams).double()
    mvalid = lib_graph.build_graph(is_training=False, hparams=hparams)

    print(m)

    # tracker = Tracker(
    #     label='validation loss',
    #     patience=FLAGS.patience,
    #     decay_op=m.decay_op,
    #     save_path=os.path.join(FLAGS.logdir, hparams.log_subdir_str,
    #                             'best_model.pt'))

    validation_loss = 0
    loss_best = 100

    epoch_count = 0
    while epoch_count < FLAGS.num_epochs or not FLAGS.num_epochs:

        # Run training.
        # run_epoch(sv, sess, m, train_data, hparams, m.train_op, 'train',
        #             epoch_count)
        data_seed = None
        with lib_util.numpy_seed(data_seed):
            batches = (
                train_data.get_featuremaps().batches(
                    size=m.batch_size, shuffle=True, shuffle_rng=data_seed
                )
            )
        losses = lib_util.AggregateMean('losses')
        losses_total = lib_util.AggregateMean('losses_total')
        losses_mask = lib_util.AggregateMean('losses_mask')
        losses_unmask = lib_util.AggregateMean('losses_unmask')

        optim = torch.optim.Adam(m.parameters(), hparams.learning_rate)
        compute_predictions = torch.nn.Sigmoid()
        cross_entropy = torch.nn.MultiLabelSoftMarginLoss()
        if hparams.use_softmax_loss:
            compute_predictions = torch.nn.Softmax(dim=2)
            cross_entropy = torch.nn.LogSoftmax(dim=2)

        start_time = time.time()
        for iteration, batch in enumerate(batches):
            optim.zero_grad()
            feed_dict = batch.get_feed_dict(m.placeholders)
            i = 0
            for key, value in feed_dict.items():
                placeholder, features = value
                if key == 'masks':
                    masks = torch.from_numpy(features).double()
                elif key == 'pianorolls':
                    pianorolls = torch.from_numpy(features).double()
                i += 1
            convnet_input = get_convnet_input(pianorolls, masks, hparams)
            # Convert to channels first format
            convnet_input = convnet_input.permute(0,3,2,1)
            pianorolls = pianorolls.permute(0,3,2,1)
            print('Batch #{}'.format(iteration), convnet_input.shape)

            prediction = compute_predictions(m(convnet_input))
            print(prediction[0].max(), pianorolls[0].max())
            loss = cross_entropy(prediction, pianorolls)
            if loss < loss_best:
                loss_best = loss
            print('Loss: {:.2f}'.format(loss))
            loss.backward()
            optim.step()


        # Run validation.
        # sv = 0
        # sess = 0
        # if epoch_count % hparams.eval_freq == 0:
        #     estimate_popstats(sv, sess, m, train_data, hparams)
        #     loss = run_epoch(sv, sess, mvalid, valid_data, hparams, no_op,
        #                         'valid', epoch_count)
        #     # tracker(loss, sess)
        #     # if tracker.should_stop():
        #     #     break

        epoch_count += 1

    print('best', 'Validation Loss', loss_best)
    print('Done.')
    return loss_best

def get_convnet_input(pianorolls, masks, hparams):
    pianorolls *= 1. - masks
    if hparams.mask_indicates_context:
        masks = 1. - masks
    return torch.cat([pianorolls, masks], dim=3)


class Tracker(object):
    """Tracks the progress of training and checks if training should stop."""

    def __init__(self, label, save_path, sign=-1, patience=5, decay_op=None):
        self.label = label
        self.sign = sign
        self.best = np.inf
        self.saver = tf.train.Saver()
        self.save_path = save_path
        self.patience = patience
        # NOTE: age is reset with decay, but true_age is not
        self.age = 0
        self.true_age = 0
        self.decay_op = decay_op

    def __call__(self, loss, sess):
        if self.sign * loss > self.sign * self.best:
            if FLAGS.log_progress:
                print('Previous best %s: %.4f.',
                                self.label, self.best)
                os.mkdir(os.path.dirname(self.save_path))
                self.saver.save(sess, self.save_path)
                print('Storing best model so far with loss %.4f at %s.' %
                                (loss, self.save_path))
            self.best = loss
            self.age = 0
            self.true_age = 0
        else:
            self.age += 1
            self.true_age += 1
            if self.age > self.patience:
                sess.run([self.decay_op])
                self.age = 0

    def should_stop(self):
        return self.true_age > 5 * self.patience


def _print_popstat_info(tfpopstats, nppopstats):
    """Prints the average and std of population versus batch statistics."""
    mean_errors = []
    stdev_errors = []
    for j, (tfpopstat, nppopstat) in enumerate(zip(tfpopstats, nppopstats)):
        moving_average = tfpopstat.eval()
        if j % 2 == 0:
            mean_errors.append(abs(moving_average - nppopstat))
        else:
            stdev_errors.append(
                abs(np.sqrt(moving_average) - np.sqrt(nppopstat)))

    def flatmean(xs):
        return np.mean(np.concatenate([x.flatten() for x in xs]))

    print('average of pop mean/stdev errors: %g %g' % (flatmean(mean_errors),
                                                       flatmean(stdev_errors)))
    print('average of batch mean/stdev: %g %g' %
          (flatmean(nppopstats[0::2]),
           flatmean([np.sqrt(ugh) for ugh in nppopstats[1::2]])))

def _hparams_from_flags():
  """Instantiate hparams based on flags set in FLAGS."""
  keys = ("""
      dataset quantization_level num_instruments separate_instruments
      crop_piece_len architecture use_sep_conv num_initial_regular_conv_layers
      sep_conv_depth_multiplier num_dilation_blocks dilate_time_only
      repeat_last_dilation_level num_layers num_filters use_residual
      batch_size maskout_method mask_indicates_context optimize_mask_only
      rescale_loss patience corrupt_ratio eval_freq run_id
      num_pointwise_splits interleave_split_every_n_layers
      """.split())
  hparams = lib_hparams.Hyperparameters(**dict(
      (key, getattr(FLAGS, key)) for key in keys))
  return hparams

if __name__ == '__main__':
    FLAGS = _hyperparams_from_flags()
    main(0, FLAGS)
