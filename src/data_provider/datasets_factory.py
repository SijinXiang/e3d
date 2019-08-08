from src.data_provider import milan

def data_provider(dataset_name,
				train_data_paths,
				valid_data_paths,
				batch_size,
				img_width,
				in_seq_length,
				out_seq_length,
				dims_3D,
				is_training):

	if dataset_name != 'milan':
		raise ValueError('Name of dataset unknown %s' % dataset_name)
        
	test_input_param = {
		'tra_set': False,
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
			'tra_set': True,
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
