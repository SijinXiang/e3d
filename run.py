# Many thanks to daya for modifying the code :)
# ==============================================================================

"""Main function to run the code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import numpy as np
from src.data_provider import datasets_factory
from src.models.model_factory import Model
import src.trainer as trainer
#from src.utils import preprocess
import tensorflow as tf
import argparse


def add_arguments(parser):
	parser.register("type", "bool", lambda v: v.lower() == "true")

	parser.add_argument("--train_data_paths", type=str, default="", help="train data paths")
	parser.add_argument("--valid_data_paths", type=str, default="", help="validation data paths")
	parser.add_argument("--test_data_paths", type=str, default="", help="train data paths")
	parser.add_argument("--save_dir", type=str, default="", help="dir to store trained net")
	parser.add_argument("--gen_frm_dir", type=str, default="", help="dir to store result.")

	parser.add_argument("--is_training", type="bool", nargs="?", const=True,
					default=False,
					help="training or testing")
	parser.add_argument("--dataset_name", type=str, default="milan", help="name of dataset")
	parser.add_argument("--input_seq_length", type=int, default=10, help="number of input snapshots")
	parser.add_argument("--output_seq_length", type=int, default=10, help="number of output snapshots")
	parser.add_argument("--dimension_3D", type=int, default=2, help="dimension of input depth")
	parser.add_argument("--img_width", type=int, default=100, help="input image width.")
	parser.add_argument("--patch_size", type=int, default=1, help="patch size on one dimension")
	parser.add_argument("--reverse_input", type="bool", nargs="?", const=True,
					default=False, 
					help="reverse the input/outputs during training.")
    
	parser.add_argument("--model_name", type=str, default="e3d_lstm", help="The name of the architecture")
	parser.add_argument("--pretrained_model", type=str, default="", help=".ckpt file to initialize from")
	parser.add_argument("--num_hidden", type=str, default="10,10,10,10", help="COMMA separated number of units of e3d lstms")
	parser.add_argument("--filter_size", type=int, default=5, help="filter of a e3d lstm layer")
	parser.add_argument("--layer_norm", type="bool", nargs="?", const=True,
					default=True, 
					help="whether to apply tensor layer norm")

	parser.add_argument("--scheduled_sampling", type="bool", nargs="?", const=True,
					default=True, 
					help="for scheduled sampling")
	parser.add_argument("--sampling_stop_iter", type=int, default=40, help="for scheduled sampling")
	parser.add_argument("--sampling_start_value", type=float, default=1.0, help="for scheduled sampling")
	parser.add_argument("--sampling_changing_rate", type=float, default=0.00002, help="for scheduled sampling")
    
	parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
	parser.add_argument("--batch_size", type=int, default=50, help="batch size for training")
	parser.add_argument("--max_iterations", type=int, default=50, help="max num of steps")
	parser.add_argument("--display_interval", type=int, default=1, help="number of iters showing training loss")
	parser.add_argument("--test_interval", type=int, default=1, help="number of iters for test")
	parser.add_argument("--snapshot_interval", type=int, default=50, help="number of iters saving models")
#	parser.add_argument("--num_save_samples", type=int, default=10, help="number of sequences to be saved")
	parser.add_argument("--n_gpu", type=int, default=1, help="how many GPUs to distribute the training across")
	parser.add_argument("--allow_gpu_growth", type="bool", nargs="?", const=True,
					default=True, 
					help="allow gpu growth")



def main(unused_argv):
	"""Main function."""
	print(FLAGS)
	# print(FLAGS.reverse_input)
	if tf.gfile.Exists(FLAGS.save_dir):
		tf.gfile.DeleteRecursively(FLAGS.save_dir)
	tf.gfile.MakeDirs(FLAGS.save_dir)
	if tf.gfile.Exists(FLAGS.gen_frm_dir):
		tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
	tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

	gpu_list = np.asarray(
		os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
	FLAGS.n_gpu = len(gpu_list)
	print('Initializing models')

	model = Model(FLAGS)

	if FLAGS.is_training:
		train_wrapper(model)
	else:
		test_wrapper(model)


def schedule_sampling(eta, itr):
	"""Gets schedule sampling parameters for training."""
	zeros = np.zeros((FLAGS.batch_size, FLAGS.output_seq_length // FLAGS.dimension_3D - 1, FLAGS.img_width, FLAGS.img_width, FLAGS.dimension_3D))
	if not FLAGS.scheduled_sampling:
		return 0.0, zeros

	if itr < FLAGS.sampling_stop_iter:
		eta -= FLAGS.sampling_changing_rate
	else:
		eta = 0.0
	random_flip = np.random.random_sample(
		(FLAGS.batch_size, FLAGS.output_seq_length // FLAGS.dimension_3D - 1))
	true_token = (random_flip < eta)
	ones = np.ones((FLAGS.img_width // FLAGS.patch_size, FLAGS.img_width // FLAGS.patch_size, FLAGS.patch_size**2*FLAGS.dimension_3D))
	zeros = np.zeros((FLAGS.img_width // FLAGS.patch_size, FLAGS.img_width // FLAGS.patch_size, FLAGS.patch_size**2 * FLAGS.dimension_3D))
	real_input_flag = []
	for i in range(FLAGS.batch_size):
		for j in range(FLAGS.output_seq_length // FLAGS.dimension_3D - 1):
			if true_token[i, j]:
				real_input_flag.append(ones)
			else:
				real_input_flag.append(zeros)
	real_input_flag = np.array(real_input_flag)
	real_input_flag = np.reshape(real_input_flag,(FLAGS.batch_size, FLAGS.output_seq_length // FLAGS.dimension_3D - 1,FLAGS.img_width // FLAGS.patch_size, FLAGS.img_width // FLAGS.patch_size,FLAGS.patch_size**2 * FLAGS.dimension_3D))
    
	return eta, real_input_flag


def train_wrapper(model):
	"""Wrapping function to train the model."""
	if FLAGS.pretrained_model:
		model.load(FLAGS.pretrained_model)
  # load data
	train_input_handle, test_input_handle = datasets_factory.data_provider(
		FLAGS.dataset_name,
		FLAGS.train_data_paths,
		FLAGS.valid_data_paths,
		FLAGS.batch_size * FLAGS.n_gpu,
		FLAGS.img_width,
		FLAGS.input_seq_length,
		FLAGS.output_seq_length,
		FLAGS.dimension_3D,
		is_training=True)

	eta = FLAGS.sampling_start_value

	tra_cost = 0.0
	batch_id = 0
	stopping = [10000000000000000]
	for itr in range(1, FLAGS.max_iterations + 1):
		if train_input_handle.no_batch_left():
			model.save(itr)
			print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'itr: ' + str(itr))
			print('training loss: ' + str(tra_cost / batch_id))
			val_cost = trainer.test(model, test_input_handle,FLAGS, itr)
			if val_cost < stopping[-1]:
				stopping = [val_cost]
			elif len(stopping) < 10:
				stopping.append(val_cost)
			if len(stopping) == 10:
				break
			train_input_handle.begin(do_shuffle=True)
			tra_cost = 0
			batch_id = 0

		ims = train_input_handle.get_batch()
		batch_id += 1

		eta, real_input_flag = schedule_sampling(eta, itr)

		tra_cost += trainer.train(model, ims, real_input_flag, FLAGS, itr)

		#if itr % FLAGS.snapshot_interval == 0:
			#model.save(itr)

		#if itr % FLAGS.test_interval == 0:
			#trainer.test(model, test_input_handle, FLAGS, itr)

		train_input_handle.next_batch()


def test_wrapper(model):
	model.load(FLAGS.pretrained_model)
	test_input_handle = datasets_factory.data_provider(
		FLAGS.dataset_name,
		FLAGS.train_data_paths, 
		FLAGS.test_data_paths, # Should use test data rather than training or validation data.
		FLAGS.batch_size * FLAGS.n_gpu,
		FLAGS.img_width,
		FLAGS.input_seq_length,
		FLAGS.output_seq_length,
		FLAGS.dimension_3D,
		is_training=False)
	trainer.test(model, test_input_handle, FLAGS, 'test_result')


if __name__ == '__main__':
	nmt_parser = argparse.ArgumentParser()
	add_arguments(nmt_parser)
	FLAGS, unparsed = nmt_parser.parse_known_args()
	tf.app.run(main=main)

