import numpy as np
import time
import sys
import psutil
import humanize
import os
import GPUtil as GPU

import train_test_individual
import init_data
# rated individual: [gene_a, gene_b,..., classification loss]
# rated population: [rated individual 0, rated individual 1, ..]


def random_gen(count: int) -> np.ndarray:
    random_numbers = np.random.rand(count)
    return random_numbers


def batch_size_incr_possible(conf):
    try:
        gpu = GPU.getGPUs()[0]
        if gpu.memoryFree > conf["min_free_gpu_memory"]:
            return True
        else:
            return False

    except:
        print("no GPU!")
        return False


def check_timeout(evo_conf) -> bool:
    now_time = time.time()
    if now_time - evo_conf["start_time"] > evo_conf["timeout_secs"]:
        return True
    else:
        return False


def end_if_timeout(evo_conf):
    if check_timeout(evo_conf):
        print("\n\n====== TIMEOUT AFTER " + str(evo_conf["timeout_secs"]) + "s ==========")
        sys.exit()


def mock_rate_loss(individual) -> float:
    # fitness = np.sum(individual)
    gene_count = individual.shape[0]
    unfitness = 0
    for i in range(gene_count):
        unfitness += (-1)*(abs(individual[gene_count-1]-0.5))
    return unfitness


def add_loss(unrated_individual: np.ndarray, train_test_data, train_test_config) -> np.ndarray:
    loss = train_test_individual.train_test_individual(unrated_individual, train_test_config, train_test_data)
    rated_individual = np.concatenate((unrated_individual, loss), axis=None)
    return rated_individual


def create_rated_individual(train_test_data, train_test_config, evo_conf) -> np.ndarray:
    end_if_timeout(evo_conf)
    individual = np.empty(evo_conf["gene_count"])
    for i in range(0, evo_conf["gene_count"]):
        individual[i] = np.random.uniform(evo_conf["gene_ranges"][i][0], evo_conf["gene_ranges"][i][1])

    rated_individual = add_loss(individual, train_test_data, train_test_config)
    return rated_individual


# rated population is already sorted , but sorting again is still required when adding new individuals
def create_rated_population(train_test_data, train_test_config, evo_conf) -> np.ndarray:

    population = np.empty((0, evo_conf["gene_count"] + 1))
    for i in range(evo_conf["population_size"]):
        individual = create_rated_individual(train_test_data, train_test_config, evo_conf)
        population = np.vstack([population, individual])

    sorted_population = population[population[:, evo_conf["gene_count"]].argsort()]
    return sorted_population


# sorts population according to fitness in desc order and returns the $count best individuals
def get_best_individuals(rated_population: np.ndarray, evo_conf) -> np.ndarray:
    evo_conf["gene_count"] = rated_population.shape[1] - 1
    sorted_population = rated_population[rated_population[:, evo_conf["gene_count"]].argsort()]

    best_individuals = np.empty((0, evo_conf["gene_count"]+1))

    for i in range(evo_conf["parent_count"]):
        ith_best_individual = sorted_population[i]
        best_individuals = np.vstack([best_individuals, ith_best_individual])
    return best_individuals


def mutate(gene: float, chance: float) -> float:
    if np.random.rand() < chance:
        gene = gene * np.random.rand()
    return gene


# combines the same trait from multiple parents
def combine_genes_rand_weight(genes: np.ndarray, rng) -> float:
    genes = genes.flatten()
    parent_count = len(genes)
    weights = rng(parent_count)
    normalized_weights = weights / np.sum(weights)
    gene_result = float(np.dot(normalized_weights, genes))
    return gene_result


def make_rated_children(parents: np.ndarray, train_test_data, train_test_config, evo_conf) -> np.ndarray:
    child = np.zeros(evo_conf["gene_count"])
    children = np.empty((0, parents.shape[1]))

    for ch in range(evo_conf["parent_count"]):
        for gene in range(evo_conf["gene_count"]):
            child[gene] = combine_genes_rand_weight(parents[:, [gene]], random_gen)

        rated_child = add_loss(child, train_test_config, train_test_data)
        children = np.vstack([children, rated_child])

    return children


def batch_size_stress_test(train_test_data, evo_conf):
    gene_ranges = evo_conf["gene_ranges"]
    individual = np.empty(evo_conf["gene_count"])

    for gene in range(0, evo_conf["gene_count"]):
        max_gene = gene_ranges[gene][-1]
        individual[gene] = max_gene

    train_test_conf_stress = {"train_epochs": 1,
                              "batch_size": 1,
                              "batch_size_max": 200,
                              "min_free_gpu_memory": 2000,
                              "log_file_path": "stress_run_log.txt",
                              "checkpoint_path": "stress_model.h5",
                              "fold_count": 2,
                              "first_val_loss_max": 2,
                              "patience": 0,
                              "train_verbose": 0}

    runs = int(np.log2(train_test_conf_stress["batch_size_max"]) + 1)
    for i in range(0, runs):
        add_loss(individual, train_test_data, train_test_conf_stress)
        if batch_size_incr_possible(train_test_conf_stress):
            print("Enough GPU mem, increasing batch size")
            train_test_conf_stress["batch_size"] = min(int(train_test_conf_stress["batch_size"] * 2),
                                                       train_test_conf_stress["batch_size_max"])
            print("Batch Size Now:" + str(train_test_conf_stress["batch_size"]))

        else:
            print("No Batch Size increase possible, trying to step back one notch for safety")
            train_test_conf_stress["batch_size"] = max(int(train_test_conf_stress["batch_size"] / 2), 1)
            print("Batch Size Now:" + str(train_test_conf_stress["batch_size"]))
            break

    return train_test_conf_stress["batch_size"]

def main():

    my_gene_ranges = np.array([[0.00001, 0.001],  # learning rate
                               [1, 1],  # feature_size
                               [1, 2],   # conv_layer_count
                               [1, 3],   # fc_layer_count
                               [1, 10],  # fc_neurons
                               [1, 2],   # kernel_size
                               [1, 1],  # dilation_rate
                               [0.0, 0.8]])  # dropout

    my_gene_ranges_dict = {"learning_rate": [0.00001, 0.001],
                           "feature_size": [1, 1],
                           "conv_layer_count": [1, 2],
                           "fc_layer_count":  [1, 3],
                           "fc_neurons": [1, 10],
                           "kernel_size": [1, 2],
                           "dilation_rate": [1, 2],
                           "dropout": [0.0, 0.8]
                           }

    evo_conf = {"epochs": 50,
                "population_size": 200,
                "gene_count": 8,
                "parent_count": 2,
                "children_count": 5,
                "gene_ranges": my_gene_ranges,
                "timeout_secs": 28800,
                "start_time": 0}

    data_conf = {"train_data_dir": "data/training_data",
                 "test_data_dir": "data/test_data",
                 "train_cut_start": 0,
                 "train_cut_length": 6000,
                 "test_cut_start": 1000,
                 "test_cut_length": 5000,
                 "aug_multiplier": 1}

    train_test_conf = {"train_epochs": 1,
                       "batch_size": 1,
                       "log_file_path": "run_log.txt",
                       "checkpoint_path": "model.h5",
                       "fold_count": 2,
                       "first_val_loss_max": 2,
                       "patience": 5,
                       "train_verbose": 0}

    evo_conf["start_time"] = time.time()
    train_test_data = init_data.init_data(data_conf)

    train_test_conf["batch_size"] = batch_size_stress_test(train_test_data, evo_conf)
    print("Batch size determined by the stress test: " + str(train_test_conf["batch_size"] ))

    my_initial_population = create_rated_population(train_test_data, train_test_conf, evo_conf)

    best_individuals = get_best_individuals(my_initial_population, evo_conf)

    for e in range(evo_conf["epochs"]):
        print("\n\n====== Evolution Epoch: " + str(e) + "==============")
        my_children = make_rated_children(best_individuals, train_test_data, train_test_conf, evo_conf)
        best_individuals = get_best_individuals(my_children, evo_conf)
        print(best_individuals[0][-1])


if __name__ == '__main__':
    main()
