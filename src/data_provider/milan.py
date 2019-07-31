# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Moving MNIST Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np


class InputHandle(object):
	"""Class for handling dataset inputs."""

	def __init__(self, input_param):
		self.paths = input_param['paths']
		#self.num_paths = len(input_param['paths'])
		self.name = input_param['name']
		self.input_data_type = input_param.get('input_data_type', 'float32')
		self.output_data_type = input_param.get('output_data_type', 'float32')
		self.minibatch_size = input_param['minibatch_size']
		self.input_seq_length = input_param['input_seq_length']
		self.output_seq_length = input_param['output_seq_length']
		self.dims_3D = input_param['3D_dims']
		#self.is_output_sequence = input_param['is_output_sequence']
		self.data = {'input_raw_data': [],
					'output_raw_data': []}
		self.indices = []
		self.current_position = 0
		self.current_batch_size = 0
		self.current_batch_indices = []
		self.current_input_length = 0
		self.load()

	def load(self):
		"""Load the data."""
		snapshots = np.load(self.paths)
		total_shots = snapshots.shape[0]
		timestep = self.input_seq_length // self.dims_3D
        
		for i in range(total_shots // (self.input_seq_length+self.output_seq_length)):
			start = int(i * (self.input_seq_length + self.output_seq_length))
			end = int((i+1) * (self.input_seq_length + self.output_seq_length))
            
			in_raw = snapshots[start:start+self.input_seq_length].reshape((timestep, self.dims_3D, 100, 100))
			out_raw = snapshots[start+self.input_seq_length:end].reshape((-1, self.dims_3D, 100, 100))
            
			self.data['input_raw_data'].append(in_raw)
			self.data['output_raw_data'].append(out_raw)

            
	'''
        for key in dat_1.keys():
            self.data[key] = dat_1[key]
        if self.num_paths == 2:
            dat_2 = np.load(self.paths[1])
            num_clips_1 = dat_1['clips'].shape[1]
            dat_2['clips'][:, :, 0] += num_clips_1
            self.data['clips'] = np.concatenate((dat_1['clips'], dat_2['clips']),
                                              axis=1)
            self.data['input_raw_data'] = np.concatenate(
              (dat_1['input_raw_data'], dat_2['input_raw_data']), axis=0)
            self.data['output_raw_data'] = np.concatenate(
              (dat_1['output_raw_data'], dat_2['output_raw_data']), axis=0)
        for key in self.data.keys():
            print(key)
            print(self.data[key].shape)'''

	def total(self):
		"""Returns the total number of clips."""
		return len(self.data['input_raw_data'])

	def begin(self, do_shuffle=True):
		"""Move to the begin of the batch."""
		self.indices = np.arange(self.total(), dtype='int32')
		if do_shuffle:
			random.shuffle(self.indices)
		self.current_position = 0
		if self.current_position + self.minibatch_size <= self.total():
			self.current_batch_size = self.minibatch_size
		else:
			self.current_batch_size = self.total() - self.current_position
		self.current_batch_indices = self.indices[self.current_position:self.current_position +self.current_batch_size]
#         self.current_input_length = max(
#             self.data['clips'][0, ind, 1] for ind in self.current_batch_indices)
#         self.current_output_length = max(
#             self.data['clips'][1, ind, 1] for ind in self.current_batch_indices)

	def next_batch(self):
		"""Move to the next batch."""
		self.current_position += self.current_batch_size
		if self.no_batch_left():
			return None
		if self.current_position + self.minibatch_size <= self.total():
			self.current_batch_size = self.minibatch_size
		else:
			self.current_batch_size = self.total() - self.current_position
		self.current_batch_indices = self.indices[self.current_position:self.current_position +self.current_batch_size]
#     self.current_input_length = max(
#         self.data['clips'][0, ind, 1] for ind in self.current_batch_indices)
#     self.current_output_length = max(
#         self.data['clips'][1, ind, 1] for ind in self.current_batch_indices)

	def no_batch_left(self):
		if self.current_position >= self.total() - self.current_batch_size:
			return True
		else:
			return False

	def input_batch(self):
		"""Processes for the input batches."""
		if self.no_batch_left():
			return None
		input_batch = np.zeros((self.current_batch_size, int(self.input_seq_length / self.dims_3D), self.dims_3D, 100, 100))
		for i in range(self.current_batch_size):
			batch_ind = self.current_batch_indices[i]
			input_batch[i] = self.data['input_raw_data'][batch_ind]
		input_batch = np.transpose(input_batch, (0, 1, 3, 4, 2))
		return input_batch

	'''
    input_batch = np.zeros((self.current_batch_size,
                            self.current_input_length) +
                           tuple(self.data['dims'][0])).astype(
                               self.input_data_type)
    input_batch = np.transpose(input_batch, (0, 1, 3, 4, 2))
    for i in range(self.current_batch_size):
      batch_ind = self.current_batch_indices[i]
      begin = self.data['clips'][0, batch_ind, 0]
      end = self.data['clips'][0, batch_ind, 0]
      end = end + self.data['clips'][0, batch_ind, 1]
      data_slice = self.data['input_raw_data'][begin:end, :, :, :]
      data_slice = np.transpose(data_slice, (0, 2, 3, 1))
      input_batch[i, :self.current_input_length, :, :, :] = data_slice
    input_batch = input_batch.astype(self.input_data_type)
    return input_batch
	'''

	def output_batch(self):
		"""Processes for the output batches."""
		if self.no_batch_left():
			return None
		output_batch = np.zeros((self.current_batch_size, int(self.output_seq_length / self.dims_3D), self.dims_3D, 100, 100))        
		for i in range(self.current_batch_size):
			batch_ind = self.current_batch_indices[i]
			output_batch[i] = self.data['output_raw_data'][batch_ind]
		output_batch = np.transpose(output_batch, (0, 1, 3, 4, 2))
		return output_batch


	def get_batch(self):
		input_seq = self.input_batch()
		output_seq = self.output_batch()
		batch = np.concatenate((input_seq, output_seq), axis=1) # How tensorflow tell which is X which is Y?
		return batch
