from Utils import *
from Genome import *


class InnovationManager:

    def __init__(self, node_idx_start: int):
        self.node_idx = node_idx_start
        self.innovation_idx = 0
        self.innovation_map = {}

    def check_for_innovation(self, connection, add_connection: bool = False):
        key = Key(connection.in_idx.id, connection.out_idx.id)
        innov = self.innovation_map.get(key)
        if innov is not None:
            return innov
        if add_connection:
            self.innovation_idx += 1
            self.innovation_map[key] = self.innovation_idx
            return self.innovation_idx
        return 0

    def compare_innovation_in_same_generation(self, gen1, gen2):
        pass

    def get_next_node_id(self):
        self.node_idx += 1
        return self.node_idx

    # def compareInnovationInSameGeneration(self, population):
    #     genomes = population.population.copy()
    #     for g1 in genomes:
    #         for g2 in genomes[1:]:
