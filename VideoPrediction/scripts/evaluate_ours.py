# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import errno
import json
import os
import random
import io
import sys
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import datasets, models, metrics

def compute_our_diversity_np(our_npy):
    # should be (num_samples, batch, generated_len, height, width, 3)
    gen_samples = np.transpose(our_npy, [1, 0, 2, 3, 4, 5])

    tensor_shape = list(gen_samples.shape)
    #print(tensor_shape)


    # now (batch, num_samples, generated_len * height * width * 3)
    gen_samples = np.reshape(gen_samples, tensor_shape[:2] + [-1])

    squared_form = np.sum(gen_samples * gen_samples, axis=2)
    # turn r into column vector
    squared_form = np.expand_dims(squared_form, -1)

    # do x^2 - 2xy + y^2
    squared_distance = squared_form - 2 * np.matmul(gen_samples, np.transpose(gen_samples, [0, 2, 1])) + np.transpose(squared_form, [0, 2, 1])
    # should include (batch, num_samples, num_samples)
    # each value in num_samples x num_samples includes the squared distance between each sample

    return np.sum(np.absolute(squared_distance)) / ( tensor_shape[0] * tensor_shape[1] * (tensor_shape[1] - 1) * tensor_shape[2] * tensor_shape[3] * tensor_shape[4] * tensor_shape[5])


def save_metrics(prefix_fname, metrics, sample_start_ind=0):
    head, tail = os.path.split(prefix_fname)
    if head and not os.path.exists(head):
        os.makedirs(head)
    assert metrics.ndim == 2
    file_mode = 'wb' if sample_start_ind == 0 else 'ab'
    with io.open('%s.csv' % prefix_fname, file_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if sample_start_ind == 0:
            writer.writerow(map(str, ['sample_ind'] + list(range(metrics.shape[1])) + ['mean']))
        for i, metrics_row in enumerate(metrics):
            writer.writerow(map(unicode, map(str, [sample_start_ind + i] + list(metrics_row) + [np.mean(metrics_row)])))


def load_metrics(prefix_fname):
    with io.open('%s.csv' % prefix_fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        rows = list(reader)
        # skip header (first row), indices (first column), and means (last column)
        metrics = np.array(rows)[1:, 1:-1].astype(np.float32)
    return metrics


def save_image_sequence(prefix_fname, images, overlaid_images=None, centers=None,
                        radius=5, alpha=0.8, time_start_ind=0):
    import cv2
    head, tail = os.path.split(prefix_fname)
    if not os.path.exists(head):
        os.makedirs(head)
    if images.shape[-1] == 1:
        images = as_heatmap(images)
    if overlaid_images is not None:
        assert images.shape[-1] == 3
        assert overlaid_images.shape[-1] == 1
        gray_images = rgb2gray(images)
        overlaid_images = as_heatmap(overlaid_images)
        images = (1 - alpha) * gray_images[..., None] + alpha * overlaid_images
    for t, image in enumerate(images):
        image_fname = '%s_%02d.png' % (prefix_fname, time_start_ind + t)
        if centers is not None:
            scale = np.max(np.array([256, 256]) / np.array(image.shape[:2]))
            image = resize_and_draw_circle(image, np.array(image.shape[:2]) * scale, centers[t], radius,
                                           edgecolor='r', fill=False, linestyle='--', linewidth=2)
        image = (image * 255.0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_fname, image)



def save_image_sequences(prefix_fname, images, overlaid_images=None, centers=None,
                         radius=5, alpha=0.8, sample_start_ind=0, time_start_ind=0):
    head, tail = os.path.split(prefix_fname)
    if head and not os.path.exists(head):
        os.makedirs(head)
    if overlaid_images is None:
        overlaid_images = [None] * len(images)
    if centers is None:
        centers = [None] * len(images)
    for i, (images_, overlaid_images_, centers_) in enumerate(zip(images, overlaid_images, centers)):
        images_fname = '%s_%05d' % (prefix_fname, sample_start_ind + i)
        save_image_sequence(images_fname, images_, overlaid_images_, centers_,
                            radius=radius, alpha=alpha, time_start_ind=time_start_ind)


def save_prediction_results(task_dir, results, model_hparams, sample_start_ind=0, only_metrics=False):
    context_frames = model_hparams.context_frames
    sequence_length = model_hparams.sequence_length
    context_images = results['context']
    images = results['images']
    gen_images = results['gen_images']
    mse = metrics.mean_squared_error_np(images, gen_images, keep_axis=(0, 1))
    save_metrics(os.path.join(task_dir, 'metrics', 'mse'),
                 mse, sample_start_ind=sample_start_ind)

    if only_metrics:
        return

    save_image_sequences(os.path.join(task_dir, 'inputs', 'context_image'),
                         context_images, sample_start_ind=sample_start_ind)
    save_image_sequences(os.path.join(task_dir, 'inputs', 'gt'),
                         images, sample_start_ind=sample_start_ind)
    save_image_sequences(os.path.join(task_dir, 'outputs', 'gen_image'),
                         gen_images, sample_start_ind=sample_start_ind)


def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]
    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.
    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return zip(*[lst[i::n] for i in range(n)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories "
                                                                     "train, val, test, etc, or a directory containing "
                                                                     "the tfrecords")
    parser.add_argument("--results_dir", type=str, default='results', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where results are saved. default is results_dir/model_fname, "
                                             "where model_fname is the directory name of checkpoint")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--only_metrics", action='store_true')

    parser.add_argument("--mode", type=str, choices=['val', 'test'], default='test', help='mode for dataset, val or test.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")

    parser.add_argument("--batch_size", type=int, default=1, help="number of samples in batch")
    parser.add_argument("--num_samples", type=int, help="number of samples in total (all of them by default)")
    parser.add_argument("--num_epochs", type=int, default=1)

    parser.add_argument("--eval_substasks", type=str, nargs='+', default=['max', 'avg', 'min'], help='subtasks to evaluate (e.g. max, avg, min). only applicable to prediction_eval')
    parser.add_argument("--num_stochastic_samples", type=int, default=100)

    parser.add_argument("--gt_inputs_dir", type=str, help="directory containing input ground truth images for ismple dataset")
    parser.add_argument("--gt_outputs_dir", type=str, help="directory containing output ground truth images for ismple dataset")

    parser.add_argument("--eval_parallel_iterations", type=int, default=1)
    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        if not os.path.isdir(args.checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.model = args.model or options['model']
        try:
            with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                dataset_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
                model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
        args.output_dir = args.output_dir or os.path.join(args.results_dir, os.path.split(checkpoint_dir)[1])
    else:
        if not args.dataset:
            raise ValueError('dataset is required when checkpoint is not specified')
        if not args.model:
            raise ValueError('model is required when checkpoint is not specified')
        args.output_dir = args.output_dir or os.path.join(args.results_dir, 'model.%s' % args.model)

    if not args.only_metrics:
        args.num_stochastic_samples = args.num_stochastic_samples // 10

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    VideoDataset = datasets.get_dataset_class(args.dataset)
    dataset = VideoDataset(args.input_dir, mode=args.mode, num_epochs=args.num_epochs, seed=args.seed,
                           hparams_dict=dataset_hparams_dict, hparams=args.dataset_hparams)

    def override_hparams_dict(dataset):
        hparams_dict = dict(model_hparams_dict)
        hparams_dict['context_frames'] = dataset.hparams.context_frames
        hparams_dict['sequence_length'] = dataset.hparams.sequence_length
        hparams_dict['repeat'] = dataset.hparams.time_shift
        hparams_dict['zs_seed_no'] = args.seed
        return hparams_dict

    VideoPredictionModel = models.get_model_class(args.model)
    model = VideoPredictionModel(mode='test', hparams_dict=override_hparams_dict(dataset), hparams=args.model_hparams,
                                 eval_num_samples=args.num_stochastic_samples, eval_parallel_iterations=args.eval_parallel_iterations)
    context_frames = model.hparams.context_frames
    sequence_length = model.hparams.sequence_length

    if args.num_samples:
        if args.num_samples > dataset.num_examples_per_epoch():
            raise ValueError('num_samples cannot be larger than the dataset')
        num_examples_per_epoch = args.num_samples
    else:
        num_examples_per_epoch = dataset.num_examples_per_epoch()
    if num_examples_per_epoch % args.batch_size != 0:
        raise ValueError('batch_size should evenly divide the dataset')

    inputs, target = dataset.make_batch(args.batch_size)
    if not isinstance(model, models.GroundTruthVideoPredictionModel):
        # remove ground truth data past context_frames to prevent accidentally using it
        for k, v in inputs.items():
            if k != 'actions':
                inputs[k] = v[:, :context_frames]


    input_phs = {k: tf.placeholder(v.dtype, [v.get_shape().as_list()[0] * 2] + v.get_shape().as_list()[1:], '%s_ph' % k) for k, v in inputs.items()}
    target_ph = tf.placeholder(target.dtype, [target.get_shape().as_list()[0] * 2] + target.get_shape().as_list()[1:], 'targets_ph')

    with tf.variable_scope(''):
        model.build_graph(input_phs, target_ph)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(dataset.hparams.values(), sort_keys=True, indent=4))
    with open(os.path.join(output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)

    model.restore(sess, args.checkpoint)

    sample_ind = 0
    our_result_list = {}
    while True:
        if args.num_samples and sample_ind >= args.num_samples:
            break
        try:
            input_results, target_result = sess.run([inputs, target])
        except tf.errors.OutOfRangeError:
            break

        print("evaluation samples from %d to %d" % (sample_ind, sample_ind + args.batch_size))

        ##########################################
        # compute statistics
        feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
        feed_dict.update({target_ph: target_result})

        ###################################
        # compute diversity measures
        fetches = {'images': model.inputs['images'],
                   'gen_images': model.outputs['gen_images']}

        all_results = [sess.run(fetches, feed_dict=feed_dict) for _ in range(args.num_stochastic_samples)]
        all_results = nest.map_structure(lambda *x: np.stack(x), *all_results)
        # the result has (num_samples, batch_size, time, height, width, 3)

        all_results['context'] = np.repeat(np.expand_dims(input_results['images'][:args.batch_size], axis=0), args.num_stochastic_samples, axis=0)
        all_results['images'] = np.repeat(np.expand_dims(target_result[:args.batch_size], axis=0), args.num_stochastic_samples, axis=0)
        all_results['gen_images'] = all_results['gen_images'][:, :args.batch_size, context_frames - sequence_length:]

        all_images = all_results['images']
        all_gen_images = all_results['gen_images']

        if args.only_metrics:
            # do it for VGG
            csim_result = [metrics.vgg_cosine_similarity_np(np.array(a), np.array(b), keep_axis=(0, 1, 2)) for a, b in zip(group(all_images, 25), group(all_gen_images, 25))]
            csim_max = np.mean(np.array(csim_result), axis=-1).max()
            our_result_list.setdefault('gt_csim', []).append(csim_max)

            # do it for MSE
            all_mse = metrics.mean_squared_error_np(all_images, all_gen_images, keep_axis=(0, 1, 2))
            mse_min = np.mean(all_mse, axis=-1).min()
            our_result_list.setdefault('gt_mse', []).append(mse_min)

            # compute ours
            our_result = compute_our_diversity_np(all_gen_images)
            our_result_list.setdefault('btw_mse', []).append(our_result)

            #all_mse_argsort = np.argsort(all_mse, axis=0)
            #for subtask, argsort_ind in zip(['_best', '_median', '_worst'],
            #                                [0, args.num_stochastic_samples // 2, -1]):
            #    all_mse_inds = all_mse_argsort[argsort_ind]
            #    gather = lambda x: np.array([x[ind, sample_ind] for sample_ind, ind in enumerate(all_mse_inds)])
            #    results = nest.map_structure(gather, all_results)
            #    save_prediction_results(os.path.join(output_dir, 'prediction' + subtask),
            #                            results, model.hparams, sample_ind, (args.only_metrics or (subtask == '_median')) )
        else:
            # write logic for saving results
            print('saving results')
            for sample_no, single_video in enumerate(all_results['gen_images']):
                save_image_sequence(os.path.join(output_dir, '%05d_%02d' % (sample_ind, sample_no)), single_video[0])

        sample_ind += args.batch_size


    summary_file_array = []
    summary_file_path = os.path.join(args.output_dir, 'summary.txt')

    for key, value_list in our_result_list.items():
        our_metric = np.asarray(value_list)
        sentence = '[%s]: %12.9f (%12.9f)' % (key, our_metric.mean(), our_metric.std())
        print(sentence)
        summary_file_array.append(sentence)

    with open(summary_file_path, 'w') as f:
        f.write('\n'.join(summary_file_array))


if __name__ == '__main__':
    main()
