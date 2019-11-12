import numpy as np
import train_test_individual
import init_data
# rated individual: [gene_a, gene_b,..., classification loss]
# rated population: [rated individual 0, rated individual 1, ..]


def mock_rate_loss(individual) -> float:
    # fitness = np.sum(individual)
    gene_count = individual.shape[0]
    unfitness = 0
    for i in range(gene_count):
        unfitness += (-1)*(abs(individual[gene_count-1]-0.5))
    return unfitness


def add_loss(unrated_individual: np.ndarray, train_test_config, train_test_data) -> np.ndarray:
    loss = train_test_individual.train_test_individual(unrated_individual, train_test_config, train_test_data)
    #loss = mock_rate_loss(unrated_individual)
    rated_individual = np.concatenate((unrated_individual, loss), axis=None)
    return rated_individual


def create_rated_individual(train_test_data, train_test_config, evo_conf) -> np.ndarray:

    individual = np.empty(evo_conf["gene_count"])
    for i in range(0, evo_conf["gene_count"]):
        individual[i] = np.random.uniform(evo_conf["gene_ranges"][i][0], evo_conf["gene_ranges"][i][1])

    rated_individual = add_loss(individual, train_test_config, train_test_data)
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


# combines the same trait from multiple parents
def combine_genes_avg(genes: np.ndarray) -> int:
    gene_result = 0
    genes = genes.flatten()
    parent_count = len(genes)  # each gene is from a different parent
    for gene in genes:
        gene_result += gene / parent_count
    return gene_result


def mutate(gene: float, chance: float) -> float:
    if np.random.rand() < chance:
        gene = gene * np.random.rand()
    return gene


# combines the same trait from multiple parents
def combine_genes_rand_weight(genes: np.ndarray) -> float:
    gene_result = 0
    genes = genes.flatten()
    parent_count = len(genes)

    weights = np.random.rand(parent_count)
    normalized_weights = weights / np.sum(weights)

    for index, gene in enumerate(genes):
        gene_result += gene * normalized_weights[index]

    gene_result = mutate(gene_result, 0.1)
    return gene_result


def make_rated_children(parents: np.ndarray, train_test_data, train_test_config, evo_conf) -> np.ndarray:
    child = np.zeros(evo_conf["gene_count"])
    children = np.empty((0, parents.shape[1]))

    for ch in range(evo_conf["parent_count"]):
        for gene in range(evo_conf["gene_count"]):
            child[gene] = combine_genes_rand_weight(parents[:, [gene]])

        rated_child = add_loss(child, train_test_config, train_test_data)
        children = np.vstack([children, rated_child])

    return children


def main():

    my_gene_ranges = np.array([[0.00001, 0.01],  # learning rate
                               [1, 1],  # feature_size
                               [1, 1],   # conv_layer_count
                               [1, 1],   # fc_layer_count
                               [10, 10],  # fc_neurons
                               [1, 2],   # kernel_size
                               [1, 100],  # dilation_rate
                               [0.0, 0.8]])  # dropout

    evo_conf = {"epochs": 5,
                "population_size": 200,
                "gene_count": 8,
                "parent_count": 2,
                "children_count": 5,
                "gene_ranges": my_gene_ranges,
                "timeout_secs": 200}

    data_conf = {"train_data_dir": "data/training_data",
                 "test_data_dir": "data/test_data",
                 "train_cut_start": 0,
                 "train_cut_length": 6000,
                 "test_cut_start": 1000,
                 "test_cut_length": 5000,
                 "aug_multiplier": 1}

    train_test_conf = {"train_epochs": 1,
                       "train_batch_size": 20,
                       "test_batch_size": 48,
                       "log_file_path": "run_log.txt",
                       "checkpoint_path": "model.h5",
                       "fold_count": 2,
                       "first_val_loss_max": 2,
                       "patience": 5,
                       "train_verbose": 0}


    #start_time = time.time()
    train_test_data = init_data.init_data(data_conf)

    my_initial_population = create_rated_population(train_test_data, train_test_conf,  evo_conf)

    best_individuals = get_best_individuals(my_initial_population, evo_conf)

    for e in range(evo_conf["epochs"]):
        print("====== Evolution Epoch: " + str(e) + "==============")
        my_children = make_rated_children(best_individuals, train_test_data, train_test_conf, evo_conf)
        best_individuals = get_best_individuals(my_children, evo_conf)
        print(best_individuals[0][-1])


if __name__ == '__main__':
    main()
