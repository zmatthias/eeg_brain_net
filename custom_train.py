import train_test_individual
import init_data

if __name__ == '__main__':

	aug_multiplier = 3

	custom_genes = [0.0001,  # learning rate
					2,  # feature_size
					2,  # conv_layer_count
					2,  # kernel_size
					1,  # dilation_rate
					0.1]  # dropout

	config_data = {"train_data_dir": "data/training_data",
						"test_data_dir": "data/test_data",
						"train_cut_start": 0,
						"train_cut_length": 6000,
						"test_cut_start": 1000,
						"test_cut_length": 5000,
						"aug_multiplier": 3}

	config_train_test = {"train_epochs": 6,
						 "train_batch_size": 5,
						 "test_batch_size": 48,
						 "log_file_path": "run_log.txt",
						 "fold_count": 5,
						 "train_verbose": 1}

	x_train, y_train, x_test, y_test = init_data.init_data(config_data)
	train_test_individual.train_test_individual(custom_genes, config_train_test, x_train, y_train, x_test, y_test)
