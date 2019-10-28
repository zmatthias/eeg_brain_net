import train_test_individual
import init_data

if __name__ == '__main__':

	custom_genes = [0.0001,  # learning rate
					50,  # feature_size
					5,  # conv_layer_count
					2,  # fc_layer_count
					200, #fc_neurons
					5,  # kernel_size
					20,  # dilation_rate
					0.1]  # dropout

	data_config = {"train_data_dir": "data/training_data",
					"test_data_dir": "data/test_data",
					"train_cut_start": 0,
					"train_cut_length": 6000,
					"test_cut_start": 1000,
					"test_cut_length": 5000,
					"aug_multiplier": 5}

	train_test_config = {"train_epochs": 1000,
						 "train_batch_size": 200,
						 "test_batch_size": 48,
						 "log_file_path": "run_log.txt",
						 "fold_count": 5,
						 "train_verbose": 1}

	train_test_data = init_data.init_data(data_config)
	train_test_individual.train_test_individual(custom_genes, train_test_config, train_test_data)
