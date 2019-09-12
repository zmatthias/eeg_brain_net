import numpy as np
import train_brain

# rated individual: [gene_a, gene_b,..., classification loss]
# rated population: [rated individual 0, rated individual 1, ..]


def mock_rate_loss(individual) -> float:
    # fitness = np.sum(individual)
    gene_count = individual.shape[0]
    unfitness = 0
    for i in range(gene_count):
        unfitness += (-1)*(abs(individual[gene_count-1]-0.5))
    return unfitness


def rate_loss(individual) -> float:
    loss = train_brain.train_test_individual(individual, x_train, y_train, x_test, y_test)
    return loss


def add_loss(unrated_individual: np.ndarray) -> np.ndarray:
    loss = rate_loss(unrated_individual)
    rated_individual = np.concatenate((unrated_individual, loss), axis=None)
    return rated_individual


def create_rated_individual(gene_count) -> np.ndarray:
    individual = np.random.rand(gene_count)
    rated_individual = add_loss(individual)
    return rated_individual


# rated population is already sorted , but sorting again is still required when adding new individuals
def create_rated_population(population_count: int, gene_count: int) -> np.ndarray:
    population = np.empty((0, gene_count + 1))
    for i in range(population_count):
        individual = create_rated_individual(gene_count)
        population = np.vstack([population, individual])

    sorted_population = population[population[:, gene_count].argsort()]
    return sorted_population


# sorts population according to fitness in desc order and returns the $count best individuals
def get_best_individuals(rated_population: np.ndarray, count: int) -> np.ndarray:
    gene_count = rated_population.shape[1] - 1
    sorted_population = rated_population[rated_population[:, gene_count].argsort()]

    best_individuals = np.empty((0, gene_count+1))

    for i in range(count):
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
    #gene_result = round(gene_result)
    return gene_result


def mutate(gene: float, chance: float) -> float:
    if np.random.rand() < chance:
        gene = gene * np.random.rand()
    return gene


# combines the same trait from multiple parents
def combine_genes_rand_weight(genes: np.ndarray) -> int:
    gene_result = 0
    genes = genes.flatten()
    parent_count = len(genes)

    weights = np.random.rand(parent_count)
    normalized_weights = weights / np.sum(weights)

    for index, gene in enumerate(genes):
        gene_result += gene * normalized_weights[index]

    gene_result = mutate(gene_result, 0.1)
    return gene_result


def make_rated_children(parents: np.ndarray, count: int) -> np.ndarray:
    gene_count = parents.shape[1] - 1  # subtract fitness score
    child = np.zeros(gene_count)
    children = np.empty((0, parents.shape[1]))

    for ch in range(count):
        for gene in range(gene_count):
            child[gene] = combine_genes_rand_weight(parents[:, [gene]])

        rated_child = add_loss(child)
        children = np.vstack([children, rated_child])

    return children


if __name__ == '__main__':
    epochs = 50
    population_size = 500
    my_gene_count = 5
    my_parent_count = 5
    my_children_count = 30

    train_data_dir = "data/training_data"
    test_data_dir = "data/test_data"
    aug_multiplier = 2

    x_train, y_train, x_test, y_test = train_brain.init_data(train_data_dir,test_data_dir,aug_multiplier)

    my_initial_population = create_rated_population(population_size, my_gene_count)
    besties = get_best_individuals(my_initial_population, my_parent_count)

    for e in range(epochs):
        my_children = make_rated_children(besties, my_children_count)
        besties = get_best_individuals(my_children, my_parent_count)
        print(besties[0][-1])
