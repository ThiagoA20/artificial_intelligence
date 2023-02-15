import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import math
import random
import logging
import global_vars
import json
from copy import deepcopy

logger = logging.getLogger(__name__)

# ----- Activation functions -------------------------------------------------------------------

def sigmoid(x: float) -> float:
    try:
        return round(1 / (1 + math.exp(-x)), 6)
    except OverflowError:
        return 0

def ReLU(x: float) -> float:
    return max(0, x)

def Step(x: float) -> float:
    if x <= 0:
        return 0.0
    else:
        return 1.0

# ----- Network Configurations -----------------------------------------------------------------

INNOVATION_NUM = 0

GENOME_HASHTABLE = {}

# ----- Implementation Details -----------------------------------------------------------------

class Neuron:

    def __init__(self, neuron_id: int, neuron_type: int, neuron_layer: int, activation: callable=sigmoid, sum_input=0.0, sum_output=0.0, active=True):
        if not isinstance(neuron_id, int) or not isinstance(neuron_type, int) or not isinstance(neuron_layer, int):
            raise TypeError("All values must be integers!")
        if not (0 <= neuron_type <= 3):
            raise ValueError("Please inform a valid neuron type!")
        self.__neuron_id = neuron_id
        self.__neuron_type = neuron_type
        self.__neuron_layer = neuron_layer
        self.__activation = activation
        self.__sum_input = sum_input
        self.__sum_output = sum_output
        self.__active = active
    
    def get_neuron_info(self) -> dict:
        return {"id": self.__neuron_id, "type": self.__neuron_type, "layer": self.__neuron_layer, "Sum result": self.__sum_input, "output": self.__sum_output, "Active": self.__active}
    
    def get_id(self) -> int:
        return self.__neuron_id
    
    def get_type(self) -> int:
        return self.__neuron_type
    
    def set_layer(self, layer: int):
        self.__neuron_layer = layer

    def get_layer(self) -> int:
        return self.__neuron_layer
    
    def calculate_sum(self, connection_results: list) -> None:
        self.__sum_input = round(sum(connection_results), 6)
    
    def get_sum(self) -> float:
        return self.__sum_input
    
    def activate_neuron(self) -> None:
        if (self.__neuron_type == 1):
            self.__sum_output = self.__sum_input
        else:
            self.__sum_output = self.__activation(self.__sum_input)
    
    def get_output(self) -> float:
        return self.__sum_output

    def change_state(self) -> None:
        self.__active = not self.__active
    
    def is_active(self) -> bool:
        return self.__active


class Connection:

    def __init__(self, innovation_number: int, in_neuron_id: int, out_neuron_id: int, weight: float=0.0, active: bool=True, recurrent: bool=False):
        if (in_neuron_id == out_neuron_id):
            raise ValueError("The value of the in and out neuron's id can't be the same!")
        if not isinstance(innovation_number, int) or not isinstance(in_neuron_id, int) or not isinstance(out_neuron_id, int) or not isinstance(weight, float) or not isinstance(active, bool):
            raise TypeError
        self.__innovation_id = innovation_number
        self.__in_neuron = in_neuron_id
        self.__out_neuron = out_neuron_id
        self.__weight = weight
        self.__active = active
        self.__is_Recurrent = recurrent

    def get_info(self) -> dict:
        return {"innovation number": self.__innovation_id, "in neuron": self.__in_neuron, "out neuron": self.__out_neuron, "weight": self.__weight, "active": self.__active, "recurrent": self.__is_Recurrent}
    
    def get_innovation_num(self) -> int:
        return self.__innovation_id

    def get_ids(self) -> tuple:
        return (self.__in_neuron, self.__out_neuron)

    def set_weight(self, weight: float):
        if not isinstance(weight, float):
            raise TypeError("The weight must be a float number!")
        self.__weight = weight
    
    def get_weight(self) -> float:
        return self.__weight
    
    def change_state(self) -> None:
        self.__active = not self.__active
    
    def is_active(self) -> bool:
        return self.__active
    
    def change_recurrency(self) -> None:
        self.__is_Recurrent = not self.__is_Recurrent
    
    def is_recurrent(self) -> bool:
        return self.__is_Recurrent


def calculate_initial_connections(ipn_amount: int, hd_amount: int, opt_amount: int, ic_percentage: int) -> int:
    real_connections = lambda x: math.ceil(x * ic_percentage/100)
    if hd_amount > 0:
        connection_amount = (ipn_amount * hd_amount) + (hd_amount * opt_amount)
    else:
        connection_amount = (ipn_amount * opt_amount)
    return real_connections(connection_amount)


def generate_initial_neuron_list(ipn_amount: int, hd_amount: int, opt_amount: int) -> list[Neuron]:
    neuron_list = []
    neuron_counter = 0
    total_neurons = ipn_amount + opt_amount + hd_amount + 1
    while neuron_counter < total_neurons:
        if neuron_counter == 0:
            neuron_list.append(Neuron(neuron_counter, 3, 1))
        else:
            if neuron_counter <= ipn_amount:
                neuron_list.append(Neuron(neuron_counter, 1, 1))
            elif neuron_counter <= ipn_amount + opt_amount:
                if not hd_amount == 0:
                    neuron_list.append(Neuron(neuron_counter, 2, 3))
                else:
                    neuron_list.append(Neuron(neuron_counter, 2, 2))
            else:
                neuron_list.append(Neuron(neuron_counter, 0, 2))
        neuron_counter += 1
    return neuron_list


def generate_initial_connection_list(total_connections: int, ipn_amount: int, hn_amount: int, opn_amount: int) -> list[Connection]:
    global GENOME_HASHTABLE, INNOVATION_NUM

    connection_list = []
    hn_start = 1 + ipn_amount + opn_amount
    opn_start = 1 + ipn_amount
    while total_connections > 0:
            if hn_amount == 0:
                for ipn in range(1, ipn_amount + 1):
                    for opn in range(opn_start, opn_start + opn_amount):
                        if f"{ipn}|{opn}" in GENOME_HASHTABLE:
                            connection_list.append(Connection(GENOME_HASHTABLE[f"{ipn}|{opn}"], ipn, opn, random.uniform(-20, 20)))
                            total_connections -= 1
                        else:
                            INNOVATION_NUM += 1
                            GENOME_HASHTABLE[f"{ipn}|{opn}"] = INNOVATION_NUM
                            connection_list.append(Connection(GENOME_HASHTABLE[f"{ipn}|{opn}"], ipn, opn, random.uniform(-20, 20)))
                            total_connections -= 1
                        if total_connections == 0:
                            break
                    if total_connections == 0:
                            break
            else:
                for hn in range(hn_start, hn_start + hn_amount):
                    for ipn in range(1, ipn_amount + 1):
                        if f"{ipn}|{hn}" in GENOME_HASHTABLE:
                            connection_list.append(Connection(GENOME_HASHTABLE[f"{ipn}|{hn}"], ipn, hn, random.uniform(-20, 20)))
                            total_connections -= 1
                        else:
                            INNOVATION_NUM += 1
                            GENOME_HASHTABLE[f"{ipn}|{hn}"] = INNOVATION_NUM
                            connection_list.append(Connection(GENOME_HASHTABLE[f"{ipn}|{hn}"], ipn, hn, random.uniform(-20, 20)))
                            total_connections -= 1
                        if total_connections == 0:
                            break
                    for opn in range(opn_start, opn_start + opn_amount):
                        if f"{hn}|{opn}" in GENOME_HASHTABLE:
                            connection_list.append(Connection(GENOME_HASHTABLE[f"{hn}|{opn}"], hn, opn, random.uniform(-20, 20)))
                            total_connections -= 1
                        else:
                            INNOVATION_NUM += 1
                            GENOME_HASHTABLE[f"{hn}|{opn}"] = INNOVATION_NUM
                            connection_list.append(Connection(GENOME_HASHTABLE[f"{hn}|{opn}"], hn, opn, random.uniform(-20, 20)))
                            total_connections -= 1
                        if total_connections == 0:
                            break
                    if total_connections == 0:
                        break
    return connection_list


class Brain:
    def __init__(self, input_neurons: int, hidden_neurons: int, output_neurons: int, connections_percentage: int, layers: dict = {}, fitness: float = 0.0, specie_num: int = None, ajusted_fitness: float = None, neuron_list: list[Neuron] = [], connection_list: list[Connection] = []):
        global GENOME_HASHTABLE, INNOVATION_NUM
        self.__input_neurons = input_neurons
        self.__hidden_neurons = hidden_neurons
        self.__output_neurons = output_neurons
        self.__connection_percentage = connections_percentage
        if neuron_list == []:
            self.__neuron_list = generate_initial_neuron_list(input_neurons, hidden_neurons, output_neurons)
        else:
            self.__neuron_list = neuron_list
        total_connections = calculate_initial_connections(input_neurons, hidden_neurons, output_neurons, connections_percentage)
        if connection_list == []:
            self.__connection_list = generate_initial_connection_list(total_connections, input_neurons, hidden_neurons, output_neurons)
        else:
            self.__connection_list = connection_list
        self.__layers = layers
        self.__fitness = fitness
        self.__specie_num = specie_num
        self.__ajusted_fitness = ajusted_fitness
    
    def get_brain_info(self) -> dict:
        return {"input neurons": self.__input_neurons, "hidden neurons": self.__hidden_neurons, "output neurons": self.__output_neurons, "connection percentage": self.__connection_percentage, "neuron list": self.__neuron_list, "connection list": self.__connection_list, "layers": self.__layers, "fitness": self.__fitness, "specie": self.__specie_num, "ajusted fitness": self.__ajusted_fitness}
    
    def load_neuron_list(self, neuron_list: list[Neuron]) -> None:
        self.__neuron_list = neuron_list

    def load_connection_list(self, connection_list: list[Connection]) -> None:
        self.__connection_list = connection_list

    def get_neuron_list(self) -> list[Neuron]:
        return self.__neuron_list
    
    def get_connection_list(self) -> list[Connection]:
        return self.__connection_list
    
    def set_layers(self, draw=False) -> None:
        neuron_list = [self.__neuron_list[neuron].get_id() for neuron in range(1, len(self.__neuron_list))]
        connection_list = [connection.get_ids() for connection in self.__connection_list]

        n_position_list = {}
        n_connection_list = {}

        for neuron in neuron_list:
            n_connection_list[str(neuron)] = []

        for connection in connection_list:
            n_connection_list[str(connection[0])].append(connection[1])
        
        camada_counter = 1
        camada_atual = []
        while len(n_position_list) < len(neuron_list):
            not_found = []
            found = []
            for neuron in n_connection_list:
                if int(neuron) not in camada_atual:
                    for n_num in n_connection_list[neuron]:
                        if n_num not in found:
                            found.append(n_num)
            for neuron in neuron_list:
                if str(neuron) not in n_position_list and neuron not in found:
                    not_found.append(neuron)
            for neuron in not_found:
                n_position_list[str(neuron)] = camada_counter
                camada_atual.append(neuron)
            camada_counter += 1
        
        for neuron in n_position_list:
            self.__neuron_list[int(neuron)].set_layer(n_position_list[neuron])
        
        if self.__layers == {} or draw:
            self.get_layers()
    
    def get_layers(self) -> dict:
        layers = {}
        for neuron in self.__neuron_list:
            neuron_id = neuron.get_id()
            if not neuron_id == 0:
                neuron_layer = str(neuron.get_layer())
                if not neuron_layer in layers:
                    layers[neuron_layer] = []
                    layers[neuron_layer].append(neuron_id)
                else:
                    layers[neuron_layer].append(neuron_id)
        self.__layers = dict(sorted(layers.items(), key=lambda item: int(item[0])))
        return dict(sorted(layers.items(), key=lambda item: int(item[0])))
    
    def load_inputs(self, input_list: list) -> None:
        if self.__layers == {}:
            self.set_layers()
        if len(input_list) != self.__input_neurons:
            raise ValueError("The number of the inputs given must be equal to the number of input neurons in the network!")
        else:
            for i in range(1, self.__input_neurons + 1):
                self.__neuron_list[i].calculate_sum([input_list[i - 1]])
                self.__neuron_list[i].activate_neuron()

    def run_network(self) -> None:
        layer_values = list(self.__layers.values())
        for layer_num in range(1, len(layer_values)):
            for neuron_num in range(len(layer_values[layer_num])):
                neuron = layer_values[layer_num][neuron_num]
                """Include the bias here"""
                input_list = []
                for connection in self.__connection_list:
                    connection_ids = connection.get_ids()
                    if connection_ids[1] == neuron and not connection.is_recurrent():
                        input_list.append(self.__neuron_list[connection_ids[0]].get_output() * connection.get_weight())
                self.__neuron_list[neuron].calculate_sum(input_list)
                self.__neuron_list[neuron].activate_neuron()
    
    def get_outputs(self) -> list:
        output_values = []
        for i in range(self.__input_neurons+1, self.__input_neurons + self.__output_neurons + 1):
            output_values.append(self.__neuron_list[i].get_output())
        return output_values

    def draw_network(self) -> None:
        self.set_layers(draw=True)

        network = [
            self.__layers,
            {f"{connection.get_ids()[0]}|{connection.get_ids()[1]}": [connection.is_active(), connection.get_weight()] for connection in self.__connection_list},
            {f"{neuron.get_id()}": [neuron.get_output()] for neuron in self.__neuron_list if neuron.get_id() != 0}
        ]

        with global_vars.network_lock:
            global_vars.network = network
    
    def set_fitness(self, fitness) -> None:
        self.__fitness += fitness
    
    def reset_fitness(self) -> None:
        self.__fitness = 0
    
    def get_fitness(self) -> float:
        return self.__fitness
    
    def set_specie(self, specie_num):
        self.__specie_num = specie_num
    
    def get_specie(self) -> int:
        return self.__specie_num
    
    def set_ajusted_fitness(self, ajusted_fitness: float) -> None:
        self.__ajusted_fitness = ajusted_fitness
    
    def get_ajusted_fitness(self) -> float:
        return self.__ajusted_fitness
    
    def set_connection_weight(self, weight_list):
        for connection in self.__connection_list:
            ids = connection.get_ids()
            if f"{ids[0]}|{ids[1]}" in weight_list:
                connection.set_weight(weight_list[f"{ids[0]}|{ids[1]}"])

    def mutate_connection_weights(self, prob):
        for connection in self.__connection_list:
            if connection.is_active():
                get_prob = random.uniform(0.0, 1.0)
                if get_prob <= prob:
                    update_value = random.uniform(0.0, 1.0)
                    if update_value <= 0.9:
                        previous_weight = connection.get_weight()
                        connection.set_weight(random.uniform(previous_weight * 0.8, previous_weight * 1.2))
                    else:
                        connection.set_weight(random.uniform(-20, 20))

    def mutate_connection_state(self, prob):
        for connection in self.__connection_list:
            get_prob = random.uniform(0.0, 1.0)
            if get_prob <= prob:
                if not connection.is_recurrent():
                    connection.change_state()

    def mutate_node_state(self, prob):
        for neuron_num in range(self.__input_neurons + self.__output_neurons + 1, len(self.__neuron_list)):
            get_prob = random.uniform(0.0, 1.0)
            if get_prob <= prob:
                neuron_id = self.__neuron_list[neuron_num].get_id()
                self.__neuron_list[neuron_num].change_state()
                for connection in self.__connection_list:
                    connection_ids = connection.get_ids()
                    if connection_ids[0] == neuron_id or connection_ids[1] == neuron_id and not connection.is_recurrent():
                        connection.change_state()

    def check_recurrent_connections(self):
        neuron_layers = {}
        for neuron in range(len(self.__neuron_list)):
            neuron_layers[f"{self.__neuron_list[neuron].get_id()}"] = self.__neuron_list[neuron].get_layer()
        
        for connection in self.__connection_list:
            connection_ids = connection.get_ids()
            if neuron_layers[f"{connection_ids[1]}"] <= neuron_layers[f"{connection_ids[0]}"]:
                if not connection.is_recurrent():
                    connection.change_recurrency()
                if connection.is_active():
                    connection.change_state()
            else:
                if connection.is_recurrent():
                    connection.change_recurrency()

    def add_connection(self, allow_recurrency=False):
        global GENOME_HASHTABLE, INNOVATION_NUM

        self.set_layers()
        new_connection = None
        attempts = 10
        while new_connection == None and attempts > 0:
            valid_n1_set = []
            for i in range(len(self.__neuron_list)):
                if i >= 1 and i <= self.__input_neurons or i > self.__input_neurons + self.__output_neurons:
                    valid_n1_set.append(i)
            if len(valid_n1_set) >= 1:
                n1 = valid_n1_set[random.randint(0, len(valid_n1_set) - 1)]
                n1_layer = self.__neuron_list[n1].get_layer()
                valid_n2_set = []
                for camada in range(n1_layer, len(self.__layers)):
                    for neuron in list(self.__layers.values())[camada]:
                        valid_n2_set.append(neuron)
                if len(valid_n2_set) >= 1:
                    if len(valid_n2_set) == 1:
                        n2 = valid_n2_set[0]
                    else:
                        n2 = valid_n2_set[random.randint(0, len(valid_n2_set) - 1)]
                    connection_exists = False
                    for connection in self.__connection_list:
                        if connection.get_ids() == (n1, n2):
                            connection_exists = True
                            break
                    if not connection_exists:
                        if f"{n1}|{n2}" not in GENOME_HASHTABLE:
                            INNOVATION_NUM += 1
                            GENOME_HASHTABLE[f"{n1}|{n2}"] = INNOVATION_NUM
                            new_connection = Connection(GENOME_HASHTABLE[f"{n1}|{n2}"], n1, n2, random.uniform(-20, 20), True)
                        else:
                            new_connection = Connection(GENOME_HASHTABLE[f"{n1}|{n2}"], n1, n2, random.uniform(-20, 20), True)
            attempts -= 1
        if new_connection != None:
            self.__connection_list.append(new_connection)

    def add_node(self):
        global GENOME_HASHTABLE, INNOVATION_NUM

        self.set_layers()
        ative_connection = False
        attempts = 5
        while not ative_connection and attempts > 0:
            if len(self.__connection_list) != 0:
                selected_connection = self.__connection_list[random.randint(0, len(self.__connection_list) - 1)]
                if selected_connection.is_active():
                    ative_connection = True
                    selected_connection.change_state()
                    neuron_id = len(self.__neuron_list)
                    new_neuron = Neuron(neuron_id, 0, 0, sigmoid)

                    connection_ids = selected_connection.get_ids()

                    if f"{connection_ids[0]}|{neuron_id}" in GENOME_HASHTABLE:
                        new_connection1 = Connection(GENOME_HASHTABLE[f"{connection_ids[0]}|{neuron_id}"], connection_ids[0], neuron_id, 1.0, True)
                    else:
                        INNOVATION_NUM += 1
                        new_connection1 = Connection(INNOVATION_NUM, connection_ids[0], neuron_id, 1.0, True)
                    
                    if f"{neuron_id}|{connection_ids[1]}" in GENOME_HASHTABLE:
                        new_connection2 = Connection(GENOME_HASHTABLE[f"{neuron_id}|{connection_ids[1]}"], neuron_id, connection_ids[1], selected_connection.get_weight(), True)
                    else:
                        INNOVATION_NUM += 1
                        new_connection2 = Connection(INNOVATION_NUM, neuron_id, connection_ids[1], selected_connection.get_weight(), True)
                    
                    self.__neuron_list.append(new_neuron)
                    self.__connection_list.append(new_connection1)
                    self.__connection_list.append(new_connection2)
            attempts -= 1
        self.set_layers()
        self.check_recurrent_connections()


class Specie:
    def __init__(self, id, individuals, fitness=0, offspring=0, gens_since_improved=0, max_fitness=0.0):
        self.id = id
        self.individuals = individuals
        self.fitness = fitness
        self.offspring = offspring
        self.gens_since_improved = gens_since_improved
        self.max_fitness = max_fitness
    
    def get_info(self) -> dict:
        return {"id": self.id, "individuals": self.individuals, "fitness": self.fitness, "gens_since_improved": self.gens_since_improved, "max_fitness": self.max_fitness}
    
    def set_max_fitness(self, max_fitness):
        self.max_fitness = max_fitness
    
    def get_max_fitness(self):
        return self.max_fitness
    
    def add_individual(self, individual_num: int) -> None:
        self.individuals.append(individual_num)
    
    def set_individuals(self, individuals: list) -> None:
        self.individuals = individuals
    
    def get_individuals_list(self) -> list[int]:
        return self.individuals
    
    def set_fitness(self, fitness: float) -> None:
        self.fitness = fitness
    
    def get_fitness(self) -> float:
        return self.fitness
    
    def set_offspring(self, offspring: int) -> None:
        self.offspring = offspring
    
    def get_offspring(self) -> int:
        return self.offspring
    
    def erase_generation(self) -> None:
        self.gens_since_improved = 0
    
    def increment_generation(self) -> None:
        self.gens_since_improved += 1


class Population:
    def __init__(self, popsize: int, brain_settings: dict, mutate_probs: dict, allow_bias: bool, allow_recurrency: bool, threshold: float = 100.0, species_target: int = 5, threshold_change_ratio: float = 0.5):
        self.__population_size = popsize
        self.__brain_settings = brain_settings
        self.__mutate_probs = mutate_probs
        self.__allow_bias = allow_bias
        self.__allow_recurrency = allow_recurrency
        self.__indivuduals_list = []
        self.__specie_list = []
        for i in range(self.__population_size):
            self.__indivuduals_list.append(Brain(brain_settings["INPUTS"], brain_settings["HIDDEN"], brain_settings["OUTPUTS"], brain_settings["CONNECTIONS"]))
        self.__generation_count = 0
        self.__threshold = threshold
        self.__threshold_change_ratio = threshold_change_ratio
        self.__species_target = species_target
        self.__pop_fitness = 0
        self.__max_fitness = 0
        self.__best_individual_id = 0
        self.__best_specie = 1
    
    def get_info(self) -> dict:
        return {
            "popsize": self.__population_size, "brain_settings": self.__brain_settings, "mutate_probs": self.__mutate_probs, "allow_bias": self.__allow_bias, "allow_recurrency": self.__allow_recurrency, "individuals_list": self.__indivuduals_list, "specie_list": self.__specie_list, "generation_count": self.__generation_count, "threshold": self.__threshold, "species_target": self.__species_target, "pop_fitness": self.__pop_fitness, "max_fitness": self.__max_fitness, "best_individual_id": self.__best_individual_id, "threshold_change_ratio": self.__threshold_change_ratio
        }

    def load_population(self, individuals_list: list[Brain], generation_count: int, pop_fitness: float, max_fitness: float, best_individual_id: int, species_list: list[Specie] = [], threshold: float = 100.0) -> None:
        self.__indivuduals_list = individuals_list
        self.__generation_count = generation_count
        self.__pop_fitness = pop_fitness
        self.__max_fitness = max_fitness
        self.__best_individual_id = best_individual_id
        self.__specie_list = species_list
        self.__threshold = threshold
    
    def save_population(self, filename: str) -> None:
        population_info = self.get_info()
        formated_individuals_list = [brain.get_brain_info() for brain in population_info['individuals_list']]
        for individual in formated_individuals_list:
            neuron_list = [neuron.get_neuron_info() for neuron in individual['neuron list']]
            individual['neuron list'] = neuron_list
            connection_list = [connection.get_info() for connection in individual['connection list']]
            individual['connection list'] = connection_list
        population_info["individuals_list"] = formated_individuals_list
        population_info["specie_list"] = []
        json_data = json.dumps(population_info, indent=4)
        with open(f"{filename}.json", 'w') as file:
            file.write(json_data)
    
    def get_pop_state(self):
        population_info = self.get_info()
        formated_individuals_list = [brain.get_brain_info() for brain in population_info['individuals_list']]
        for individual in formated_individuals_list:
            neuron_list = [neuron.get_neuron_info() for neuron in individual['neuron list']]
            individual['neuron list'] = neuron_list
            connection_list = [connection.get_info() for connection in individual['connection list']]
            individual['connection list'] = connection_list
        population_info["individuals_list"] = formated_individuals_list
        population_info["specie_list"] = []
        json_data = json.dumps(population_info, indent=4)
        return json_data

    def load_from_file(self, filename: str) -> None:
        with open(f"{filename}.json", 'r') as file:
            json_data = json.load(file)
        self.__population_size = json_data['popsize']
        self.__brain_settings = json_data['brain_settings']
        self.__mutate_probs = json_data['mutate_probs']
        self.__allow_bias = json_data['allow_bias']
        self.__allow_recurrency = json_data['allow_recurrency']
        self.__generation_count = json_data['generation_count']
        self.__threshold = json_data['threshold']
        self.__species_target = json_data['species_target']
        self.__pop_fitness = json_data['pop_fitness']
        self.__max_fitness = json_data['max_fitness']
        self.__best_individual_id = json_data['best_individual_id']
        self.__threshold_change_ratio = json_data['threshold_change_ratio']
        self.__indivuduals_list = []
        for individual in json_data['individuals_list']:
            neuron_list = [Neuron(neuron['id'], neuron['type'], neuron['layer'], sum_input=neuron['Sum result'], sum_output=neuron['output'], active=neuron['Active']) for neuron in individual['neuron list']]
            connection_list = [Connection(connection['innovation number'], connection['in neuron'], connection['out neuron'], connection['weight'], connection['active'], connection['recurrent']) for connection in individual['connection list']]
            brain = Brain(individual['input neurons'], individual['hidden neurons'], individual['output neurons'], individual['connection percentage'], neuron_list=neuron_list, connection_list=connection_list)
            self.__indivuduals_list.append(brain)

    def set_inputs(self, input_list: list[int]) -> None:
        for individual in self.__indivuduals_list:
            individual.load_inputs(input_list)
    
    def run_simulation(self) -> None:
        for individual in self.__indivuduals_list:
            individual.run_network()
    
    def calculate_fitness(self, fitness_function: callable, answers: list[int]) -> None:
        for individual in range(len(self.__indivuduals_list)):
            fitness_value = fitness_function(self.__indivuduals_list[individual].get_outputs(), answers)
            self.__indivuduals_list[individual].set_fitness(fitness_value)
    
    def get_fitness(self) -> list:
        fitness_list = []
        for individual in range(len(self.__indivuduals_list)):
            fitness_list.append(self.__indivuduals_list[individual].get_fitness())
        return fitness_list

    def draw_fittest_network(self) -> None:
        self.__indivuduals_list[self.__best_individual_id].draw_network()
        network_info = {
            "individuals": len(self.__indivuduals_list),
            "species": len(self.__specie_list),
            "generation": self.__generation_count,
            "best_individual": self.__best_individual_id,
            "best_fitness": self.__max_fitness,
            "threshold": self.__threshold,
            "best_specie": self.__best_specie,
            "connection_weight": self.__mutate_probs['connection_weight'] * 100,
            "add_connection": self.__mutate_probs['add_connection'] * 100,
            "add_node": self.__mutate_probs['add_node'] * 100,
            "connection_state": self.__mutate_probs['connection_state'] * 100,
            "node_state": self.__mutate_probs['node_state'] * 100,
            "allow_bias": self.__allow_bias,
            "allow_recurrency": self.__allow_recurrency,
            "input_neurons": self.__brain_settings['INPUTS'],
            "hidden_neurons": self.__brain_settings['HIDDEN'],
            "output_neurons": self.__brain_settings['OUTPUTS']
        }
        with global_vars.network_info_lock:
            global_vars.network_info = network_info
    
    def compare_individuals(self, individual_num: int, nay_individual: int) -> object:
        """
        Returns an object with the number of disjoint, excess, weight mean of the common connections and the genome size of the biggest parent. Only active connections are taken into account, because the are more important.
        """
        result = {"excess": 0, "disjoint": 0, "genome_size": 0, "weight_mean": 0}
        connections1 = self.__indivuduals_list[individual_num].get_connection_list()
        c1_innovation = {connection.get_innovation_num(): connection.get_weight() for connection in connections1 if connection.is_active()}
        connections2 = self.__indivuduals_list[nay_individual].get_connection_list()
        c2_innovation = {connection.get_innovation_num(): connection.get_weight() for connection in connections2 if connection.is_active()}
        if len(c1_innovation) == 0 or len(c2_innovation) == 0:
            return {"excess": 1, "disjoint": 1, "genome_size": 1, "weight_mean": 1}
        else:
            max_c1 = max(c1_innovation.keys())
            max_c2 = max(c2_innovation.keys())
            common = []
            disjoint = {}
            excess = {}
            for connection in c1_innovation:
                if connection in c2_innovation:
                    common.append(abs(c1_innovation[connection] - c2_innovation[connection]))
                else:
                    if connection < max_c2:
                        disjoint[connection] = c1_innovation[connection]
                    else:
                        excess[connection] = c1_innovation[connection]
            for connection in c2_innovation:
                if connection not in c1_innovation:
                    if connection < max_c1:
                        disjoint[connection] = c2_innovation[connection]
                    else:
                        excess[connection] = c2_innovation[connection]
            result["excess"] = len(excess)
            result["disjoint"] = len(disjoint)
            result["genome_size"] = max(len(c1_innovation), len(c2_innovation))
            if len(common) == 0:
                result["weight_mean"] = 1.0
            else:
                result["weight_mean"] = sum(common) / len(common)
            return result

    def calculate_ajusted_fitness(self) -> None:
        """Set the specie fitness and update the generations_since_improved"""
        for specie in self.__specie_list:
            individuals_list = specie.get_info()["individuals"]
            specie_fitness = 0
            for individual in individuals_list:
                ajusted_fitness = self.__indivuduals_list[individual].get_fitness() / len(individuals_list)
                self.__indivuduals_list[individual].set_ajusted_fitness(ajusted_fitness)
                specie_fitness += ajusted_fitness
            specie_fitness = specie_fitness / len(individuals_list)
            specie.set_fitness(specie_fitness)
            # if self.__generation_count == 0:
                # specie.set_fitness(specie_fitness)
            # else:
                # if specie_fitness > specie.get_max_fitness():
                    # specie.erase_generation()
                    # specie.set_fitness(specie_fitness)
                # else:
                    # specie.increment_generation()
                    # specie.set_fitness(specie_fitness)
    
    def calculate_pop_fitness(self) -> None:
        species_values = []
        for specie in self.__specie_list:
            specie_info = specie.get_info()
            species_values.append(specie_info["fitness"] * len(specie_info["individuals"]))
        self.pop_fitness = sum(species_values) / self.__population_size
    
    def set_offspring(self) -> None:
        for specie in self.__specie_list:
            specie_info = specie.get_info()
            if specie_info["gens_since_improved"] >= 15 and self.__best_individual_id not in specie_info["individuals"]:
                specie.set_offspring(0)
            else:
                specie.set_offspring((specie_info["fitness"] / self.pop_fitness) * len(specie_info["individuals"]))
            # print(f"{specie.get_info()['id']} : {specie.get_info()['gens_since_improved']} : {specie.get_offspring()}")
    
    def update_threshold(self) -> None:
        """
        Whike the number of species are lower than the threshold, it's value will decrease every generation in function of the threshold_change_ration, but if the number of species is greater, then it will increase.
        """
        num_species = len(self.__specie_list)
        if num_species < self.__species_target:
            self.__threshold -= self.__threshold_change_ratio
        elif num_species > self.__species_target:
            self.__threshold += self.__threshold_change_ratio
    
    def set_species_max_fitness(self):
        for specie in self.__specie_list:
            individuals_info = {}
            for individual in specie.get_info()["individuals"]:
                individuals_info[f"{individual}"] = self.__indivuduals_list[individual].get_fitness()
            atual_max_fitness = max(list(individuals_info.values()))
            if specie.get_max_fitness() < atual_max_fitness:
                specie.erase_generation()
            else:
                specie.increment_generation()
            specie.set_max_fitness(max(atual_max_fitness, specie.get_max_fitness()))

    def set_best_individual(self) -> None:
        fittest_id = self.__best_individual_id
        max_fitness = self.__max_fitness
        best_specie = self.__best_specie
        for individual in range(len(self.__indivuduals_list)):
            individual_fitness = self.__indivuduals_list[individual].get_fitness()
            if individual_fitness > max_fitness:
                max_fitness = individual_fitness
                fittest_id = individual
                best_specie = self.__indivuduals_list[individual].get_specie()
        self.__best_individual_id = fittest_id
        self.__max_fitness = max_fitness
        self.__best_specie = best_specie

    def speciation(self, c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> None:
        """CD = c1 * E/N + c2 * D/N + c3 * W"""
        species = 0
        species_assigned = []
        not_assigned_yet = [x for x in range(len(self.__indivuduals_list))]
        if self.__generation_count == 0:
            while len(species_assigned) < len(self.__indivuduals_list):
                species += 1
                not_assigned_yet = [x for x in not_assigned_yet if x not in species_assigned]
                new_specie = Specie(species, [])
                individual_num = not_assigned_yet[random.randint(0, len(not_assigned_yet) - 1)]
                new_specie.add_individual(individual_num)
                species_assigned.append(individual_num)
                self.__indivuduals_list[individual_num].set_specie(species)
                    
                for nay_individual in not_assigned_yet:
                    if not nay_individual == individual_num:
                        result = self.compare_individuals(individual_num, nay_individual)
                        CD = c1 * (result["excess"] / result["genome_size"]) + c2 * (result["disjoint"] / result["genome_size"]) + c3 * result["weight_mean"]
                        if CD <= self.__threshold:
                            new_specie.add_individual(nay_individual)
                            species_assigned.append(nay_individual)
                            self.__indivuduals_list[nay_individual].set_specie(species)

                self.__specie_list.append(new_specie)
        else:
            new_species = []
            for specie in self.__specie_list:
                if specie.get_offspring() != 0:
                    species += 1
                    specie_info = specie.get_info()
                    choosen_one = specie_info["individuals"][random.randint(0, len(specie_info["individuals"]) - 1)]
                    species_assigned.append(choosen_one)
                    new_specie = Specie(id=species, individuals=[choosen_one], fitness=specie_info["fitness"], gens_since_improved=specie_info["gens_since_improved"], max_fitness=specie_info['max_fitness'])
                    new_species.append(new_specie)
            self.__specie_list = new_species

            # Verificar se os outros indivídos da população se encaixam nas espécies já existentes
            for specie in self.__specie_list:
                not_assigned_yet = [x for x in not_assigned_yet if x not in species_assigned]
                choosen_one = specie.get_individuals_list()[0]
                for nay_individual in not_assigned_yet:
                    result = self.compare_individuals(choosen_one, nay_individual)
                    CD = c1 * (result["excess"] / result["genome_size"]) + c2 * (result["disjoint"] / result["genome_size"]) + c3 * result["weight_mean"]
                    if CD <= self.__threshold:
                        specie.add_individual(nay_individual)
                        species_assigned.append(nay_individual)
                        self.__indivuduals_list[nay_individual].set_specie(specie.get_info()['id'])

            # Criar novas espécies conforme for necessário
            while len(species_assigned) < len(self.__indivuduals_list):
                species += 1
                not_assigned_yet = [x for x in not_assigned_yet if x not in species_assigned]
                new_specie = Specie(species, [])
                individual_num = not_assigned_yet[random.randint(0, len(not_assigned_yet) - 1)]
                new_specie.add_individual(individual_num)
                species_assigned.append(individual_num)
                self.__indivuduals_list[individual_num].set_specie(species)
                
                for nay_individual in not_assigned_yet:
                    if not nay_individual == individual_num:
                        result = self.compare_individuals(individual_num, nay_individual)
                        CD = c1 * (result["excess"] / result["genome_size"]) + c2 * (result["disjoint"] / result["genome_size"]) + c3 * result["weight_mean"]
                        if CD <= self.__threshold:
                            new_specie.add_individual(nay_individual)
                            species_assigned.append(nay_individual)
                            self.__indivuduals_list[nay_individual].set_specie(species)

                self.__specie_list.append(new_specie)
        self.set_species_max_fitness()
        self.calculate_ajusted_fitness()
        self.calculate_pop_fitness()
        self.set_offspring()
        self.update_threshold()
        self.set_best_individual()
    
    def get_species(self) -> list[dict]:
        list_species = []
        for specie in self.__specie_list:
            list_species.append(specie.get_info())
        return list_species
    
    def get_best_individual_info(self) -> list:
        return [self.__best_individual_id, self.__indivuduals_list[self.__best_individual_id].get_fitness()]
    
    def get_best_individual_layers(self) -> dict:
        return self.__indivuduals_list[self.best_individual_id].get_layers()

    def get_species_objects(self) -> list[Specie]:
        return self.__specie_list

    def pickOne(self, individuals_info: dict, sum_fitness: float):
        individual_id = -1
        r = random.uniform(0, sum_fitness)
        list_keys = list(individuals_info.keys())
        list_values = list(individuals_info.values())
        while r >= 0:
            individual_id += 1
            r -= list_values[individual_id]
        return list_keys[individual_id]
    
    def crossover(self):
        new_individuals = []
        for i in range(self.__population_size):
            new_individuals.append(Brain(self.__brain_settings["INPUTS"], self.__brain_settings["HIDDEN"], self.__brain_settings["OUTPUTS"], self.__brain_settings["CONNECTIONS"]))
        crossover_count = 0

        total_offspring = 1

        remaining_offspring = {}
        
        for specie in self.__specie_list:
            individuals_info = {}
            for individual in specie.get_info()["individuals"]:
                individuals_info[f"{individual}"] = self.__indivuduals_list[individual].get_fitness()
            sum_fitness = sum(list(individuals_info.values()))
            
            specie_offspring = specie.get_offspring()

            offspring = int(specie_offspring)
            if f"{self.__best_individual_id}" in individuals_info:
                offspring -= 1
                new_individuals[self.__best_individual_id] = deepcopy(self.__indivuduals_list[self.__best_individual_id])
            total_offspring += offspring

            remaining_offspring[specie.get_info()['id']] = specie_offspring - offspring

            i = 0
            while i < offspring:
                crossover_count += 1
                if not crossover_count == self.__best_individual_id:
                    parent1 = self.pickOne(individuals_info, sum_fitness)
                    parent2 = self.pickOne(individuals_info, sum_fitness)
                    if individuals_info[f"{parent1}"] > individuals_info[f"{parent2}"]:
                        new_individuals[crossover_count] = deepcopy(self.__indivuduals_list[int(parent1)])
                    elif individuals_info[f"{parent1}"] == individuals_info[f"{parent2}"]:
                        selected = random.randint(1, 2)
                        if selected == 1:
                            new_individuals[crossover_count] = deepcopy(self.__indivuduals_list[int(parent1)])
                        else:
                            new_individuals[crossover_count] = deepcopy(self.__indivuduals_list[int(parent2)])
                    else:
                        new_individuals[crossover_count] = deepcopy(self.__indivuduals_list[int(parent2)])
                    
                    p1_connections = self.__indivuduals_list[int(parent1)].get_connection_list()
                    c1_values = {f"{connection.get_ids()[0]}|{connection.get_ids()[1]}": connection.get_weight() for connection in p1_connections}
                    p2_connections = self.__indivuduals_list[int(parent2)].get_connection_list()
                    c2_values = {f"{connection.get_ids()[0]}|{connection.get_ids()[1]}": connection.get_weight() for connection in p2_connections}

                    c1_common = {connection: c1_values[connection] for connection in c1_values if connection in c2_values}
                    c2_common = {connection: c2_values[connection] for connection in c2_values if connection in c1_values}
                    common = {}
                    for connection in c1_common:
                        selected_weight = random.randint(1, 2)
                        if selected_weight == 1:
                            common[connection] = c1_common[connection]
                        else:
                            common[connection] = c2_common[connection]

                    new_individuals[crossover_count].set_connection_weight(common)
                i += 1
        # print(crossover_count)
        if total_offspring != self.__population_size:
            remaining_individuals = self.__population_size - total_offspring
            sum_remaining_offspring = sum(list(remaining_offspring.values()))
            while remaining_individuals > 0:
                selected_specie = self.pickOne(remaining_offspring, sum_remaining_offspring)

                for specie in self.__specie_list:
                    if specie.get_info()['id'] == selected_specie:
                        individuals_info = {}
                        for individual in specie.get_info()["individuals"]:
                            individuals_info[f"{individual}"] = self.__indivuduals_list[individual].get_fitness()
                        sum_fitness = sum(list(individuals_info.values()))
                        
                        parent1 = self.pickOne(individuals_info, sum_fitness)
                        parent2 = self.pickOne(individuals_info, sum_fitness)
                        if individuals_info[f"{parent1}"] > individuals_info[f"{parent2}"]:
                            # print(f"CC: {crossover_count} | P1: {int(parent1)} | NIL: {len(new_individuals)} | ILL: {len(self.__indivuduals_list)}")
                            new_individuals[crossover_count] = deepcopy(self.__indivuduals_list[int(parent1)])
                        elif individuals_info[f"{parent1}"] == individuals_info[f"{parent2}"]:
                            selected = random.randint(1, 2)
                            if selected == 1:
                                # print(f"CC: {crossover_count} | P1: {int(parent1)} | NIL: {len(new_individuals)} | ILL: {len(self.__indivuduals_list)}")
                                new_individuals[crossover_count] = deepcopy(self.__indivuduals_list[int(parent1)])
                            else:
                                # print(f"CC: {crossover_count} | P1: {int(parent2)} | NIL: {len(new_individuals)} | ILL: {len(self.__indivuduals_list)}")
                                new_individuals[crossover_count] = deepcopy(self.__indivuduals_list[int(parent2)])
                        else:
                            # print(f"CC: {crossover_count} | P1: {int(parent2)} | NIL: {len(new_individuals)} | ILL: {len(self.__indivuduals_list)}")
                            new_individuals[crossover_count] = deepcopy(self.__indivuduals_list[int(parent2)])
                        
                        p1_connections = self.__indivuduals_list[int(parent1)].get_connection_list()
                        c1_values = {f"{connection.get_ids()[0]}|{connection.get_ids()[1]}": connection.get_weight() for connection in p1_connections}
                        p2_connections = self.__indivuduals_list[int(parent2)].get_connection_list()
                        c2_values = {f"{connection.get_ids()[0]}|{connection.get_ids()[1]}": connection.get_weight() for connection in p2_connections}

                        c1_common = {connection: c1_values[connection] for connection in c1_values if connection in c2_values}
                        c2_common = {connection: c2_values[connection] for connection in c2_values if connection in c1_values}
                        common = {}
                        for connection in c1_common:
                            selected_weight = random.randint(1, 2)
                            if selected_weight == 1:
                                common[connection] = c1_common[connection]
                            else:
                                common[connection] = c2_common[connection]

                        new_individuals[crossover_count].set_connection_weight(common)
                
                crossover_count += 1
                remaining_individuals -= 1
                total_offspring += 1
        
        for individual in new_individuals:
            individual.reset_fitness()

        self.__indivuduals_list = new_individuals
        self.__generation_count += 1
    
    def mutate(self):
        for individual in range(len(self.__indivuduals_list)):
            if individual != self.__best_individual_id:
                self.__indivuduals_list[individual].mutate_connection_weights(self.__mutate_probs["connection_weight"])

                add_connection_prob = random.uniform(0.0, 1.0)
                if add_connection_prob <= self.__mutate_probs["add_connection"]:
                    self.__indivuduals_list[individual].add_connection(self.__allow_recurrency)
                
                add_node_prob = random.uniform(0.0, 1.0)
                if add_node_prob <= self.__mutate_probs["add_node"]:
                    self.__indivuduals_list[individual].add_node()
                
                self.__indivuduals_list[individual].mutate_connection_state(self.__mutate_probs["connection_state"])

                # self.__indivuduals_list[individual].mutate_node_state(self.__mutate_probs["node_state"])

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
#     logger.debug(f"Genome connections hashtable: {GENOME_HASHTABLE}")

def neurons_to_csv(filename):
    pass

def connections_to_csv(filename):
    pass

"""
O id dos neurônios não importa, por que em esscência, o que estamos buscando é
a informação da melhor composição da rede e essa informação já está nas conexões,
um neurônio pode mudar de posição, mas o que importa é saber se deve haver um novo
neurônio em dada posição e quais conexões esse neurônio deve ter.

Se pensarmos em uma rede neural como uma estrutura que correlaciona fenômenos de causa e efeito no universo, podemos inferir que achariamos a velocidade da luz colocando o valor de energia nos inputs e massa nos outputs, então toda aquela abstração no meio poderia ser resumida a um número que é a velocidade da luz ao quadrado, a seleção natural iria eliminar as entradas irrelevantes se começarmos com todos os neurônios desconectados e sem neurônios escondidos.
---

- [X] Especiação
- [X] Crossover
- [X] Mutação
- [X] Threads
- [ ] Corrigir os bugs
  - [X] Mais de uma camada de saída sendo criada
  - [X] Mutação de adicionar conexões
  - [X] Fitness incorreto
  - [X] Espécies
  - [X] Threshold
  - [X] Computar apenas conexões e neurônios ativos
  - [X] Mutação de desativar conexões
  - [X] Mutação de desativar neurônios
  - [X] Conservar a melhor rede encontrada até o momento
  - [ ] Estabilizar threshold
- [ ] Bias
- [ ] Resolver problema XOR
- [ ] Testes de performance
- [ ] Conexões recorrentes
- [ ] Começar com 0 conexões entre inputs e outputs

- Fatos interessantes:
O cérebro humano possui 86 bilhões de neurônios e consome cerca de 500 kcal (2093.4KJ) por dia
Um computador convencional consome em média 576KJ de energia se deixar ele ligado por 24h
Os seres humanos tem aproximadamente de 20.000 a 23.000 genes
"""