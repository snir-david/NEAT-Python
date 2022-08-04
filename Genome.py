import copy

from Innovation import *
from ActivationFunction import *
from Phoneme import Phoneme

INPUT = "input"
OUTPUT = "output"
HIDDEN = "hidden"


def map_parents_innovation(gen1, gen2):
    gen1_innov = gen1.innovation_keys.items()
    gen2_innov = gen2.innovation_keys.items()
    intersection = gen1_innov & gen2_innov
    difference = gen1_innov ^ gen2_innov
    return intersection, difference


""" Genome Class
this class is the genotype of the Neat Chromosome. 
Sub-Classes is - NodeGen represent a node, ConnectionGen- represent connection between nodes.
"""


class Genome:

    def __init__(self, input_num: int, output_num: int, innovation_manager, init=True):
        # nodes
        self.input_node_list = []
        self.output_node_list = []
        self.num_of_inputs_node = input_num
        self.num_of_outputs_node = output_num
        self.node_gens = {}
        # connection
        self.connection_gens = {}
        self.fitness = 0
        # innovation
        self.innovation_keys = {}
        self.innovation_manager = innovation_manager
        # initialize genome
        if init:
            self.init_minimal_gen(input_num, output_num)

    def init_minimal_gen(self, input_num, output_num):
        # init input nodes and connect them to the output node
        for i in range(1, input_num + 1):
            g_in = NodeGens(i, INPUT)
            self.node_gens[i] = g_in
            self.input_node_list.append(g_in)
        # init output node
        for j in range(input_num + 1, input_num + output_num + 1):
            g_out = NodeGens(j, OUTPUT)
            self.node_gens[j] = g_out
            self.output_node_list.append(g_out)
        for input in self.input_node_list:
            for output in self.output_node_list:
                conn = ConnectionGens(self.node_gens.get(input.id), self.node_gens.get(output.id),
                                      np.random.uniform(0, 1), True, self.innovation_manager)
                self.connection_gens[conn.innovation_idx] = conn
                self.innovation_keys[Key(input.id, output.id)] = conn.innovation_idx

    def calc_fitness(self, data_x, data_y):
        # create phoneme from the genome
        fitness = 0
        phoneme = Phoneme(self)
        phoneme.create_net()
        # iterate through the input data and run it through the network
        for x, label in zip(data_x, data_y):
            # reset node value and eval on each iteration
            for g in self.node_gens.values():
                g.node_eval = 0
                g.value = 0

            phoneme.activate(x)
            output = []
            for out_node in self.output_node_list:
                output.append(out_node.node_eval)
            y = softmax(output)
            y = np.argmax(y)

            if label == y:
                # count correct classifications
                fitness += 1

        fitness /= len(data_x)
        self.fitness = fitness * 100

    def mutate(self, prob):
        mutation_idx = random.randint(0, len(MUTATION_FUNCTION) - 1)
        MUTATION_FUNCTION[mutation_idx](self, prob)


def uniform_crossover(gen1: Genome, gen2: Genome, crossover_rate):
    connection = {}
    fittest = None
    intersection, difference = map_parents_innovation(gen1, gen2)
    offspring = copy.deepcopy(gen1)
    for innovation in intersection:
        if np.random.uniform(0, 1) < 0.5:
            connection[innovation[1]] = gen1.connection_gens.get(innovation[1])
        else:
            connection[innovation[1]] = gen2.connection_gens.get(innovation[1])
    if gen1.fitness > gen2.fitness:
        fittest = gen1
    elif gen2.fitness > gen1.fitness:
        fittest = gen2
    for innovation in difference:
        tmp = fittest.connection_gens.get(innovation[1])
        if tmp is not None:
            connection[innovation[1]] = tmp
    offspring.connection_gens = connection
    return offspring


def mutate_add_noise(gen: Genome, prob):
    size = len(gen.connection_gens)
    # create bit array with 1 and 0, 1 with prob
    bit_array = np.random.choice([0, 1], size=size, p=(1 - prob, prob))
    # creating noise matrix
    noise_array = np.random.normal(0, 0.01, size=size)
    # using multiplication and bit array, saving only prob places noise
    mul = bit_array * noise_array
    # adding noise to weight
    keys = gen.connection_gens.keys()
    i = 0
    for k in gen.connection_gens:
        gen.connection_gens[k].weight += mul[i]
        i += 1
    print(gen.connection_gens)


def mutate_add_node(gen: Genome, connection):
    # create new node, connect it with in and out node
    node = NodeGens(gen.innovation_manager.get_next_node_id(), HIDDEN)
    conn_in = ConnectionGens(connection.in_idx, node, connection.weight, True, gen.innovation_manager)
    conn_out = ConnectionGens(node, connection.out_idx, 1, True, gen.innovation_manager)
    # add to lists
    gen.node_gens[node.id] = node
    gen.connection_gens[conn_in.innovation_idx] = conn_in
    gen.connection_gens[conn_out.innovation_idx] = conn_out
    gen.innovation_keys[Key(conn_in.in_idx.id, conn_in.out_idx.id)] = conn_in.innovation_idx
    gen.innovation_keys[Key(conn_out.in_idx.id, conn_out.out_idx.id)] = conn_out.innovation_idx
    # disable original connection
    connection.enabled = False


def mutate_add_connection(gen: Genome, in_node, out_node):
    conn = ConnectionGens(in_node, out_node, np.random.normal(0, 1), True, gen.innovation_manager)
    if gen.innovation_manager.check_for_innovation(conn) != 0:
        gen.connection_gens[conn.innovation_idx] = conn


class NodeGens:
    activation_functions = ActivationFunctionSet()

    def __init__(self, idx: int, node_type: str):
        self.id = idx
        self.type = node_type  # input, hidden, output
        self.activation_func = None
        self.in_edges = []
        self.out_edges = []
        self.node_eval = 0
        self.value = 0
        if node_type == HIDDEN:
            self.activation_func = NodeGens.activation_functions.get_random_function()


class ConnectionGens:
    def __init__(self, in_idx: NodeGens, out_idx: NodeGens, weight: float, is_enabled: bool, innovation_manager):
        self.in_idx = in_idx
        self.out_idx = out_idx
        in_idx.out_edges.append(self)
        out_idx.in_edges.append(self)
        self.weight = weight
        self.enabled = is_enabled
        self.innovation_idx = innovation_manager.check_for_innovation(self, add_connection=True)


MUTATION_FUNCTION = [mutate_add_connection, mutate_add_node, mutate_add_noise]
