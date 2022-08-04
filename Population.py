from Innovation import *
from Genome import *


class Population:

    def __init__(self, elite: int, population_size: int, input_size: int, output_size: int, init: bool = False):
        self.population_size = population_size
        self.population = []
        # for each genome
        self.inputs_node = input_size
        self.outputs_node = output_size
        # innovation
        self.innovation_manager = InnovationManager()
        if init:
            self.init_population()

    def init_population(self):
        for i in range(self.population_size):
            self.population.append(Genome(self.inputs_node, self.outputs_node, self.innovation_manager))

    def __get_elitism(self, amount):
        elites = []
        fitness_list = []
        # list of the genome's fitness
        for genome in self.population:
            fitness_list.append(genome.fitness)
        fitness_list = np.array(fitness_list)
        # get amount of best genomes
        while amount:
            best_gen_idx = fitness_list.argmax()
            elites.append(fitness_list[best_gen_idx])
            fitness_list[best_gen_idx] = -1  # ignore this genome in next iteration
            amount -= 1

        return elites

    def calc_fitness(self, data_x, data_y):
        for i,genome in enumerate(self.population):
            genome.calc_fitness(data_x, data_y)
            # print(f'genome {i} is done, fitness: ', genome.fitness)

    def create_new_generation(self, elitism=2):
        new_pop = self.__get_elitism(elitism)

        # choose parents with most common genes?
        # mutate elites?


