from Genome import *
from Innovation import *
from Data import *
from Phoneme import Phoneme
from matplotlib import pyplot as plt
from Population import Population

INPUT_SIZE = 1024
OUTPUT_SIZE = 10
POPULATION_SIZE = 10
ELITE_SIZE = int(POPULATION_SIZE * 0.2)
MUTATION_RATE = 1
CROSSOVER_RATE = 0.8
NUMBER_OF_GENERATIONS = 100
DECAY = 1
SAVE_BEST_MODEL = False
NORMALIZE_INPUT = False

def new_gen(current_gen: Population, train_data):
    next_gen = Population(ELITE_SIZE, POPULATION_SIZE, INPUT_SIZE)
    new_population = current_gen.get_elites(ELITE_SIZE).copy()
    parents = current_gen.roulette_select(POPULATION_SIZE - ELITE_SIZE)
    for j in range(int((POPULATION_SIZE - ELITE_SIZE))):
        offspring = uniform_crossover(parents[j], parents[j + 1], crossover_rate=CROSSOVER_RATE)
        offspring.mutate()
        new_population.append(offspring)
    next_gen.set_population(new_population)
    next_gen.calc_fitness(train_data)
    return next_gen



if __name__ == '__main__':
    # Data loading
    dataset = Data()
    dataset.set_datasets('datasets/train.csv', 'datasets/test.csv', 'datasets/validate.csv')
    # Initialize Population
    innovationManager = InnovationManager(INPUT_SIZE + OUTPUT_SIZE)
    g1 = Genome(1024, 10, innovationManager)
    g2 = Genome(1024, 10, innovationManager)

    g1.calc_fitness(dataset.validation_x, dataset.validation_y)
    g2.calc_fitness(dataset.validation_x, dataset.validation_y)

    print(g1.fitness)
    print(g2.fitness)
    mutate_add_node(g1, g1.connection_gens[1])
    uniform_crossover(g1, g2, 1)
    # phoneme = Phoneme(g1)
    # phoneme.create_net()
    # phoneme.activate(np.arange(1, 4))

    # check_mutation()

    # print()
    # print(innovationManager.innovation_map.keys())
    # print(innovationManager.innovation_idx)
    # mapParentsInnovation(g1, g2)
