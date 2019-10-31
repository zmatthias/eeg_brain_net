import train_test_individual
import init_data


def main():
	custom_genes = [0.0006856390109323057,  # learning rate
					1,  # feature_size
					2,  # conv_layer_count
					1,  # fc_layer_count
					10, #fc_neurons
					2,  # kernel_size
					16,  # dilation_rate
					0.46]  # dropout

	data_config = {"train_data_dir": "data/test_data",
					"test_data_dir": "data/test_data",
					"train_cut_start": 0,
					"train_cut_length": 6000,
					"test_cut_start": 1000,
					"test_cut_length": 5000,
					"aug_multiplier": 2}

	train_test_config = {"train_epochs": 10,
						 "train_batch_size": 5,
						 "test_batch_size": 1,
						 "log_file_path": "run_log.txt",
						 "checkpoint_path": "model.h5",
						 "fold_count": 2,
						 "train_verbose": 1,
						 }

	train_test_data = init_data.init_data(data_config)
	train_test_individual.train_test_individual(custom_genes, train_test_config, train_test_data)


if __name__ == '__main__':
	main()
