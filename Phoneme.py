from Genome import *

INPUT = "input"
OUTPUT = "output"
HIDDEN = "hidden"


class Phoneme:
    def __init__(self, genome):
        self.genome = genome
        self.layers = []

    def create_net(self):
        input_n = self.genome.input_node_list
        nodes_to_eval = set(input_n)
        finish = False
        current_layer = []
        next_layer = set()
        while not finish:
            for node in nodes_to_eval:
                for connection in node.out_edges:
                    if connection.enabled:
                        current_layer.append((connection.in_idx, connection))
                        next_layer.add(connection.out_idx)
            self.layers.append(current_layer.copy())
            finish = True
            for node in next_layer:
                if node.type != OUTPUT:
                    finish = False
                    nodes_to_eval = next_layer.copy()
                    current_layer.clear()
                    next_layer.clear()
                    break

    def activate(self, inputs):
        # assign input values
        for i in range(len(inputs)):
            self.layers[0][i][0].value = inputs[i]
        # feed forward
        next_nodes = set()
        for layer in self.layers:
            for tuple in layer:
                tuple[1].out_idx.node_eval += tuple[0].value * tuple[1].weight
                next_nodes.add(tuple[1].out_idx)
            for node in next_nodes:
                if node.type != OUTPUT:
                    node.value = node.activation_func[1](node.node_eval)
            next_nodes.clear()
