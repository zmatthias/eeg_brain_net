import train_test_individual
import init_data

if __name__ == '__main__':

	aug_multiplier = 3

	custom_genes = [2,  # feature_size
					2,  # conv_layer_count
					2,  # kernel_size
					1,  # dilation_rate
					1]  # dropout

	config_data_init = {"train_data_dir": "data/training_data",
						"test_data_dir": "data/test_data",
						"train_cut_start": 0,
						"train_cut_length": 6000,
						"test_cut_start": 1000,
						"test_cut_length": 5000,
						"aug_multiplier": 3}

	x_train, y_train, x_test, y_test = init_data.init_data(config_data_init)
	train_test_individual.train_test_individual(custom_genes, x_train, y_train, x_test, y_test)
