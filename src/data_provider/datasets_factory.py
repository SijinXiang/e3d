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

'''Data Provider.'''

# from src.data_provider import kth_action
from src.data_provider import milan

def data_provider(dataset_name,
				train_data_paths,
				valid_data_paths,
				batch_size,
				img_width,
				in_seq_length,
				out_seq_length,
				dims_3D,
				is_training=True):
	'''Returns a Dataset.

  Args:
    dataset_name: String, the name of the dataset.
    train_data_paths: List, [train_data_path1, train_data_path2...]
    valid_data_paths: List, [val_data_path1, val_data_path2...]
    batch_size: Int, the batch size.
    img_width: Int, the width of input images.
    in_seq_length, number of input snapshots.
    out_seq_length, number of output snapshots.
    dims_3D, dimension of depth channel of 3D encoder
    is_training: Bool, training or testing.

  Returns:
      if is_training is True, it returns two dataset instances for both
      training and evaluation. Otherwise only one dataset instance for
      evaluation.
  Raises:
      ValueError: When `dataset_name` is unknown.
	'''

	if dataset_name != 'milan':
		raise ValueError('Name of dataset unknown %s' % dataset_name)
        
	test_input_param = {
		'paths': valid_data_paths,
		'minibatch_size': batch_size,
		'input_data_type': 'float32',
		'input_seq_length': in_seq_length,
		'output_seq_length': out_seq_length,
		'3D_dims': dims_3D,
		'name': dataset_name + 'test iterator'
		}
                
	test_input_handle = milan.InputHandle(test_input_param)
	test_input_handle.begin(do_shuffle=False)
    
	if is_training:
		train_input_param = {
			'paths': train_data_paths,
			'minibatch_size': batch_size,
			'input_data_type': 'float32',
			'input_seq_length': in_seq_length,
			'output_seq_length': out_seq_length,
			'3D_dims': dims_3D,
			'name': dataset_name + ' train iterator'
		}
    
		train_input_handle = milan.InputHandle(train_input_param)
		train_input_handle.begin(do_shuffle=True)
		return train_input_handle, test_input_handle
	else:
		return test_input_handle

'''
  if dataset_name == 'action':
    input_param = {
        'paths': valid_data_list,
        'image_width': img_width,
        'minibatch_size': batch_size,
        'seq_length': seq_length,
        'input_data_type': 'float32',
        'name': dataset_name + ' iterator'
    }
    input_handle = datasets_map[dataset_name].DataProcess(input_param)
    if is_training:
      train_input_handle = input_handle.get_train_input_handle()
      train_input_handle.begin(do_shuffle=True)
      test_input_handle = input_handle.get_test_input_handle()
      test_input_handle.begin(do_shuffle=False)
      return train_input_handle, test_input_handle
    else:
      test_input_handle = input_handle.get_test_input_handle()
      test_input_handle.begin(do_shuffle=False)
      return test_input_handle
'''
