import train_brain

if __name__ == '__main__':

	train_data_dir = "data/training_data"
	test_data_dir = "data/test_data"
	aug_multiplier = 2

	custom_config = [0.1, 0.1, 0.1, 0.1, 0.1]

	x_train, y_train, x_test, y_test = train_brain.init_data(train_data_dir,test_data_dir,aug_multiplier)
	train_brain.train_test_individual(custom_config, x_train, y_train, x_test, y_test)
