import argparse
import glob
import itertools
import os
import random
from collections import OrderedDict

import numpy as np
from scipy import misc
import tensorflow as tf
import torchfile

from base_dataset import VarLenFeatureVideoDataset
from tensorflow.core.example import example_pb2

CONTEXT_FRAMES = 2
PREDICT_FRAMES = 12

class KTHVideoDataset(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(KTHVideoDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'images/encoded', (64, 64, 3)


    def get_default_hparams_dict(self):
        default_hparams = super(KTHVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=CONTEXT_FRAMES,
            sequence_length=CONTEXT_FRAMES + PREDICT_FRAMES,
            force_time_shift=True,
            use_state=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return False

    def num_examples_per_epoch(self):
        return len(self.filenames)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def save_tf_record(output_fname, sequences):
    #print('saving sequences to %s' % output_fname)
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for sequence in sequences:
            num_frames = len(sequence)
            height, width, channels = sequence[0].shape
            encoded_sequence = [image.tostring() for image in sequence]
            features = tf.train.Features(feature={
                'sequence_length': _int64_feature(num_frames),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'images/encoded': _bytes_list_feature(encoded_sequence),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


def read_videos_and_save_tf_records(root_dir, output_dir, meta_files, partition):
    for class_name, vids in meta_files.items():
        for vid_idx, single_video in enumerate(vids):
            str_vid_folder = single_video['vid']
            for seq_idx, single_list in enumerate(single_video['files']):
                for skip_speed in range(1, 4):
                    for init_pos in range(skip_speed):
                        new_list = single_list[init_pos::skip_speed]
                        if (len(new_list) < 20) or (partition == 'test' and (len(new_list) < 30)):
                            continue
                        output_fname = os.path.join(output_dir, '%s-%d-%d-%d-%d.tfrecords' % (class_name, vid_idx, seq_idx, skip_speed, init_pos))
                        single_seq = []
                        for file_name in new_list:
                            fname = os.path.join(root_dir, class_name, str_vid_folder, file_name)
                            im = misc.imread(fname)
                            single_seq.append(im)
                        save_tf_record(output_fname, [single_seq])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs='?', type=str, help="directory containing the directories "
                                                    "boxing, handclapping, handwaving, "
                                                    "jogging, running, walking",
                        default="./data/kth/processed/")
    parser.add_argument("process_dir", nargs='?', type=str, help="path for the t7 files",
                        default="./data/kth/processed/")
    parser.add_argument("output_dir", nargs='?', type=str, default="./data/kth/")
    args = parser.parse_args()

    partition_names = ['train', 'test'] # val would be come out later
    classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    image_size = 64

    data = {'val': {}}
    for partition in partition_names:
        data[partition] = {}
        for c in classes:
            meta_file = torchfile.load('%s/%s/%s_meta%dx%d.t7' % (args.process_dir, c, partition, image_size, image_size))
            if partition == 'train':
                random.shuffle(meta_file)
                pivot = int(0.9 * len(meta_file))
                data['val'][c] = meta_file[pivot:]
                meta_file = meta_file[:pivot]
            data[partition][c] = meta_file

    partition_names.append('val')
    for partition in partition_names:
        partition_dir = os.path.join(args.output_dir, partition)
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)
        read_videos_and_save_tf_records(args.input_dir, partition_dir, data[partition], partition)


if __name__ == '__main__':
    main()
