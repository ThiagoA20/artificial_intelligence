import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import math
import random
import logging
import global_vars

logger = logging.getLogger(__name__)

# ----- Activation functions -------------------------------------------------------------------

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0

def ReLU(x):
    pass

def Step(x):
    pass

def SoftMax(x):
    pass

# ----- Network Configurations -----------------------------------------------------------------

POPSIZE = 0
INPUT_NEURONS = 2
HIDDEN_NEURONS = 0
OUTPUT_NEURONS = 1
CONNECTIONS_PERCENTAGE = 25
DEFAULT_ACTIVATION = sigmoid

INNOVATION_NUM = 0

GENOME_HASHTABLE = {}

# ----- Implementation Details -----------------------------------------------------------------

class Neuron:

    def __init__(self, neuron_id: int, neuron_type: int, neuron_layer: int, activation: callable=sigmoid):
        if not isinstance(neuron_id, int) or not isinstance(neuron_type, int) or not isinstance(neuron_layer, int):
            raise TypeError("All values must be integers!")
        if not (0 <= neuron_type <= 3):
            raise ValueError("Please inform a valid neuron type!")
        self.__neuron_id = neuron_id
        self.__neuron_type = neuron_type
        self.__neuron_layer = neuron_layer
        self.__activation = activation
        self.__sum_input = 0
        self.__sum_output = 0
        self.__active = True
    
    def get_neuron_info(self) -> dict:
        return {"id": self.__neuron_id, "type": self.__neuron_type, "layer": self.__neuron_layer, "Sum result": self.__sum_input, "Activation result": self.__sum_output}

    def get_type(self) -> int:
        return self.__neuron_type
    
    def get_id(self) -> int:
        return self.__neuron_id
    
    def get_layer(self) -> int:
        return self.__neuron_layer
    
    def set_layer(self, layer: int):
        self.__neuron_layer = layer
    
    def get_sum(self) -> float:
        return self.__sum_input
    
    def get_output(self) -> float:
        return self.__sum_output
    
    def calculate_sum(self, connection_results: list):
        self.__sum_input = sum(connection_results)
    
    def change_state(self):
        self.__active = not self.__active
    
    def is_active(self):
        return self.__active
    
    def activate_neuron(self):
        if (self.__neuron_type == 1):
            self.__sum_output = self.__sum_input
            # self.__sum_input = 0
        else:
            self.__sum_output = self.__activation(self.__sum_input)
            # self.__sum_input = 0


class Connection:

    def __init__(self, innovation_number: int, in_neuron_id: int, out_neuron_id: int, weight: float=0.0, active: bool=True):
        if (in_neuron_id == out_neuron_id):
            raise ValueError("The value of the in and out neuron's id can't be the same!")
        if not isinstance(innovation_number, int) or not isinstance(in_neuron_id, int) or not isinstance(out_neuron_id, int) or not isinstance(weight, float) or not isinstance(active, bool):
            raise TypeError
        self.__innovation_id = innovation_number
        self.__in_neuron = in_neuron_id
        self.__out_neuron = out_neuron_id
        self.__weight = weight
        self.__active = active
        if (in_neuron_id > out_neuron_id):
            self.__is_Recurrent = True
        else:
            self.__is_Recurrent = False
    
    def get_info(self) -> dict:
        return {"innovation number": self.__innovation_id, "in neuron": self.__in_neuron, "out neuron": self.__out_neuron, "weight": self.__weight, "active": self.__active, "recurrent": self.__is_Recurrent}
    
    def get_ids(self) -> tuple:
        return (self.__in_neuron, self.__out_neuron)
    
    def get_weight(self) -> float:
        return self.__weight
    
    def is_active(self) -> bool:
        return self.__active

    def is_recurrent(self):
        return self.__is_Recurrent
    
    def change_recurrency(self):
        self.__is_Recurrent = not self.__is_Recurrent

    def set_weight(self, weight: float):
        if not isinstance(weight, float):
            raise TypeError("The weight must be a float number!")
        self.__weight = weight
    
    def get_innovation_num(self):
        return self.__innovation_id
    
    def change_state(self):
        self.__active = not self.__active


def calculate_initial_connections(ipn_amount, hd_amount, opt_amount, ic_percentage) -> int:
    real_connections = lambda x: math.ceil(x * ic_percentage/100)
    if hd_amount > 0:
        connection_amount = (ipn_amount * hd_amount) + (hd_amount * opt_amount)
    else:
        connection_amount = (ipn_amount * opt_amount)
    return real_connections(connection_amount)


def generate_initial_neuron_list(ipn_amount, hd_amount, opt_amount) -> list[Neuron]:
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
    def __init__(self, input_neurons: int = INPUT_NEURONS, hidden_neurons: int = HIDDEN_NEURONS, output_neurons: int = OUTPUT_NEURONS, connections_percentage: int = CONNECTIONS_PERCENTAGE):
        """Set the innovation number"""
        logger.debug(f"NN info: ipn = {input_neurons}; hn = {hidden_neurons}; opn = {output_neurons}; ic% = {connections_percentage}")
        global GENOME_HASHTABLE, INNOVATION_NUM

        self.specie_num = 0
        self.fitness = 0
        self.ajusted_fitness = 0

        self.__input_neurons = input_neurons
        self.__output_neurons = output_neurons

        self.__neuron_list = generate_initial_neuron_list(input_neurons, hidden_neurons, output_neurons)
        logger.debug(f"Total neurons: {len(self.__neuron_list)}")

        total_connections = calculate_initial_connections(input_neurons, hidden_neurons, output_neurons, connections_percentage)
        logger.debug(f"Total connections: {total_connections}")

        self.__connection_list = generate_initial_connection_list(total_connections, input_neurons, hidden_neurons, output_neurons)

        self.__layers = {}

        neuron_numbers = [(neuron.get_id(), neuron.get_type()) for neuron in self.__neuron_list]
        logger.debug(f"Neuron numbers/type: {neuron_numbers}")
    
    def set_layers(self, draw=False): 
        layers = {}
        connection_ids_list = []
        for connection in self.__connection_list:
            connection_ids = connection.get_ids()
            connection_ids_list.append(connection_ids)
            if not str(connection_ids[0]) in layers:
                layers[str(connection_ids[0])] = 1
            if not str(connection_ids[1]) in layers:
                layers[str(connection_ids[1])] = layers[str(connection_ids[0])] + 1
            else:
                layers[str(connection_ids[1])] = max(layers[str(connection_ids[0])] + 1, layers[str(connection_ids[1])])
        for neuron in self.__neuron_list:
            neuron_id = str(neuron.get_id())
            if not neuron_id == '0':
                neuron.set_layer(layers[neuron_id])
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
        self.__layers = dict(sorted(layers.items()))
        return dict(sorted(layers.items()))
    
    def draw_network(self):
        """
        - [ ] Layers --------- neuron numbers
        - [X] Connections ---- is_active, weight
        {'1|4': [False, 3.251], '1|5': [True, 5.123], '5|4': [True, 2.152], '2|4': [True, 1.321], '4|3': [True, 1.742]
        - [X] Neurons -------- output
        {'1': [1], '2': [1], '3': [0], '4': [0], '5': [0]}
        """
        self.set_layers(draw=True)

        network = [
            self.__layers,
            {f"{connection.get_ids()[0]}|{connection.get_ids()[1]}": [connection.is_active(), connection.get_weight()] for connection in self.__connection_list},
            {f"{neuron.get_id()}": [neuron.get_output()] for neuron in self.__neuron_list if neuron.get_id() != 0}
        ]

        with global_vars.network_lock:
            global_vars.network = network

    def get_neuron_list(self) -> list[Neuron]:
        return self.__neuron_list
    
    def get_connection_list(self) -> list[Connection]:
        return self.__connection_list
    
    def load_inputs(self, input_list: list):
        if self.__layers == {}:
            self.set_layers()
        if len(input_list) != self.__input_neurons:
            raise ValueError("The number of the inputs given must be equal to the number of input neurons in the network!")
        else:
            for i in range(1, self.__input_neurons + 1):
                self.__neuron_list[i].calculate_sum([input_list[i - 1]])
                self.__neuron_list[i].activate_neuron()

    def run_network(self):
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
    
    def set_connection_weight(self, weight_list):
        for connection in self.__connection_list:
            ids = connection.get_ids()
            if f"{ids[0]}|{ids[1]}" in weight_list:
                connection.set_weight(weight_list[f"{ids[0]}|{ids[1]}"])
    
    def get_outputs(self) -> list:
        output_values = []
        for i in range(self.__input_neurons+1, self.__input_neurons + self.__output_neurons + 1):
            output_values.append(self.__neuron_list[i].get_output())
        # print(output_values)
        return output_values

    def mutate_connection_weights(self, prob):
        for connection in self.__connection_list:
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
        while new_connection == None or attempts > 0:
            valid_n1_set = []
            for i in range(len(self.__neuron_list)):
                if i >= 1 and i <= self.__input_neurons or i > self.__input_neurons + self.__output_neurons:
                    valid_n1_set.append(i)
            n1 = valid_n1_set[random.randint(0, len(valid_n1_set) - 1)]
            n1_layer = self.__neuron_list[n1].get_layer()
            valid_n2_set = []
            for camada in range(n1_layer, len(self.__layers)): ########################## Talvez seja necessário adicionar +1 na camada para que não crie uma conexão recorrente
                for neuron in list(self.__layers.values())[camada]:
                    valid_n2_set.append(neuron)
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

    def set_fitness(self, fitness):
        self.fitness += fitness
    
    def get_fitness(self) -> float:
        return self.fitness
    
    def set_ajusted_fitness(self, ajusted_fitness):
        self.ajusted_fitness = ajusted_fitness
    
    def get_ajusted_fitness(self):
        return self.ajusted_fitness
    
    def set_specie(self, specie_num):
        self.specie_num = specie_num
    
    def get_specie(self) -> int:
        return self.specie_num


class Specie:
    def __init__(self, id, individuals, fitness=0, offspring=0, gens_since_improved=0):
        self.id = id
        self.individuals = individuals
        self.fitness = fitness
        self.offspring = offspring
        self.gens_since_improved = gens_since_improved
    
    def add_individual(self, individual_num):
        self.individuals.append(individual_num)
    
    def set_individuals(self, individuals: list):
        self.individuals = individuals
    
    def set_fitness(self, fitness):
        self.fitness = fitness
    
    def get_fitness(self):
        return self.fitness
    
    def set_offspring(self, offspring):
        self.offspring = offspring
    
    def get_offspring(self):
        return self.offspring
    
    def erase_generation(self):
        self.gens_since_improved = 0
    
    def increment_generation(self):
        self.gens_since_improved += 1
    
    def get_info(self):
        return {"id": self.id, "individuals": self.individuals, "fitness": self.fitness, "gens_since_improved": self.gens_since_improved}


class Population:
    def __init__(self, popsize: int, brain_settings: dict, mutate_probs: dict, allow_bias: bool, allow_recurrency: bool):
        self.__population_size = popsize
        self.__brain_settings = brain_settings
        self.__mutate_probs = mutate_probs
        self.__allow_bias = allow_bias
        self.__allow_recurrency = allow_recurrency
        self.__indivuduals_list = []
        self.__specie_list = []
        for i in range(self.__population_size):
            self.__indivuduals_list.append(Brain(brain_settings["INPUTS"], brain_settings["HIDDEN"], brain_settings["OUTPUTS"], brain_settings["CONNECTIONS"]))
        self.generation_count = 0
        self.threshold = 4.0
        self.pop_fitness = 0
        self.max_fitness = 0
        self.best_individual_id = 0
    
    def set_inputs(self, input_list):
        for individual in self.__indivuduals_list:
            individual.load_inputs(input_list)
    
    def run_simulation(self):
        for individual in self.__indivuduals_list:
            individual.run_network()
    
    def calculate_fitness(self, fitness_function: callable, answers):
        for individual in range(len(self.__indivuduals_list)):
            fitness_value = fitness_function(self.__indivuduals_list[individual].get_outputs(), answers)
            self.__indivuduals_list[individual].set_fitness(fitness_value)

    def draw_fittest_network(self):
        self.__indivuduals_list[self.best_individual_id].draw_network()
        network_info = {
            "individuals": len(self.__indivuduals_list),
            "species": len(self.__specie_list),
            "generation": self.generation_count,
            "best_individual": self.best_individual_id,
            "best_fitness": self.max_fitness,
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

    def get_fitness(self) -> list:
        fitness_list = []
        for individual in range(len(self.__indivuduals_list)):
            fitness_list.append({f"{individual}": self.__indivuduals_list[individual].get_fitness()})
        return fitness_list
    
    def compare_individuals(self, individual_num, nay_individual) -> object:
        """
        Retorna objeto contendo o número de disjoint, excess, média dos pesos das conexões em comum e o tamanho do genoma do parente maior
        """
        result = {"excess": 0, "disjoint": 0, "genome_size": 0, "weight_mean": 0}
        connections1 = self.__indivuduals_list[individual_num].get_connection_list()
        c1_innovation = {connection.get_innovation_num(): connection.get_weight() for connection in connections1}
        connections2 = self.__indivuduals_list[nay_individual].get_connection_list()
        c2_innovation = {connection.get_innovation_num(): connection.get_weight() for connection in connections2}
        common = []
        disjoint = []
        for connection in c1_innovation:
            if connection in c2_innovation:
                common.append(c1_innovation[connection])
            else:
                disjoint.append(c1_innovation[connection])
        for connection in c2_innovation:
            if connection not in c1_innovation and connection not in disjoint:
                disjoint.append(c2_innovation[connection])
        result["disjoint"] = len(disjoint)
        result["weight_mean"] = sum(common) / len(common)
        result["genome_size"] = max(len(connections1), len(connections2))
        if len(connections1) == result["genome_size"]:
            result["excess"] = result["genome_size"] - len(connections2)
        else:
            result["excess"] = result["genome_size"] - len(connections1)
        
        return result

    def speciation(self):
        if self.generation_count == 0:
            species = 0
            species_assigned = []
            not_assigned_yet = [x for x in range(len(self.__indivuduals_list))]
            while len(species_assigned) < len(self.__indivuduals_list):
                species += 1
                not_assigned_yet = [x for x in range(len(self.__indivuduals_list)) if x not in species_assigned]
                new_specie = Specie(species, [])
                individual_num = not_assigned_yet[random.randint(0, len(not_assigned_yet) - 1)]
                new_specie.add_individual(individual_num)
                species_assigned.append(individual_num)
                self.__indivuduals_list[individual_num].set_specie(species)
                
                for nay_individual in not_assigned_yet:
                    if not nay_individual == individual_num:
                        # CD = c1 * E/N + c2 * D/N + c3 * W
                        result = self.compare_individuals(individual_num, nay_individual)
                        CD = (result["excess"] / result["genome_size"]) + (result["disjoint"] / result["genome_size"]) + result["weight_mean"]
                        if CD <= self.threshold:
                            new_specie.add_individual(nay_individual)
                            species_assigned.append(nay_individual)
                            self.__indivuduals_list[nay_individual].set_specie(species)

                self.__specie_list.append(new_specie)
        else:
            species = 0
            species_assigned = []
            new_species = []
            for specie in self.__specie_list:
                if specie.get_offspring() != 0:
                    species += 1
                    individuals = specie.get_info()
                    choosen_one = individuals["individuals"][random.randint(0, len(individuals["individuals"]) - 1)]
                    species_assigned.append(choosen_one)
                    create_new_specie = Specie(species, [choosen_one], individuals["fitness"], individuals["gens_since_improved"])
                    new_species.append(create_new_specie)
            self.__specie_list = new_species
            
            species = 0
            not_assigned_yet = [x for x in range(len(self.__indivuduals_list))]
            while len(species_assigned) < len(self.__indivuduals_list):
                species += 1
                not_assigned_yet = [x for x in range(len(self.__indivuduals_list)) if x not in species_assigned]
                new_specie = Specie(species, [])
                individual_num = not_assigned_yet[random.randint(0, len(not_assigned_yet) - 1)]
                new_specie.add_individual(individual_num)
                species_assigned.append(individual_num)
                self.__indivuduals_list[individual_num].set_specie(species)
                
                for nay_individual in not_assigned_yet:
                    if not nay_individual == individual_num:
                        # CD = c1 * E/N + c2 * D/N + c3 * W
                        result = self.compare_individuals(individual_num, nay_individual)
                        CD = (result["excess"] / result["genome_size"]) + (result["disjoint"] / result["genome_size"]) + result["weight_mean"]
                        if CD <= self.threshold:
                            new_specie.add_individual(nay_individual)
                            species_assigned.append(nay_individual)
                            self.__indivuduals_list[nay_individual].set_specie(species)

                self.__specie_list.append(new_specie)

        for specie in self.__specie_list:
            individuals = specie.get_info()["individuals"]
            specie_fitness = 0
            for individual in individuals:
                ajusted_fitness = self.__indivuduals_list[individual].get_fitness() / len(individuals)
                self.__indivuduals_list[individual].set_ajusted_fitness(ajusted_fitness)
                specie_fitness += ajusted_fitness
            specie_fitness = specie_fitness / len(individuals)
            if self.generation_count == 0:
                specie.set_fitness(specie_fitness)
            else:
                if specie_fitness > specie.get_fitness():
                    specie.erase_generation()
                    specie.set_fitness(specie_fitness)
                else:
                    specie.increment_generation()
                    specie.set_fitness(specie_fitness)
        
        species_values = []
        for specie in self.__specie_list:
            individuals = specie.get_info()
            species_values.append(individuals["fitness"] * len(individuals["individuals"]))
        self.pop_fitness = sum(species_values) / self.__population_size

        for specie in self.__specie_list:
            individuals = specie.get_info()
            if individuals["gens_since_improved"] >= 15:
                specie.set_offspring(0)
            else:
                specie.set_offspring((individuals["fitness"] / self.pop_fitness) * len(individuals["individuals"]))
        
        fittest_id = 0
        max_fitness = 0
        for individual in range(len(self.__indivuduals_list)):
            individual_fitness = self.__indivuduals_list[individual].get_fitness()
            if individual_fitness > max_fitness:
                max_fitness = individual_fitness
                fittest_id = individual
        self.best_individual_id = fittest_id
        self.max_fitness = max_fitness
    
    def get_species(self):
        list_species = []
        for specie in self.__specie_list:
            list_species.append(specie.get_info())
        return list_species

    def pickOne(self, individuals_info: dict, sum_fitness: float):
        individual_id = -1
        r = random.uniform(0, sum_fitness)
        list_keys = list(individuals_info.keys())
        list_values = list(individuals_info.values())
        while r > 0:
            individual_id += 1
            r -= list_values[individual_id]
        return list_keys[individual_id]
    
    def crossover(self):
        new_individuals = []
        for i in range(self.__population_size):
            new_individuals.append(Brain(self.__brain_settings["INPUTS"], self.__brain_settings["HIDDEN"], self.__brain_settings["OUTPUTS"], self.__brain_settings["CONNECTIONS"]))

        crossover_count = 0

        # print(f"Individuals: {len(self.__indivuduals_list)}")
        # print(f"Species: {len(self.__specie_list)}")

        for specie in self.__specie_list:
            individuals_info = {}
            for individual in specie.get_info()["individuals"]:
                individuals_info[f"{individual}"] = self.__indivuduals_list[individual].get_fitness()
            sum_fitness = 0
            for element in individuals_info:
                sum_fitness += individuals_info[element]

            offspring = int(specie.get_offspring())
            if f"{self.best_individual_id}" in individuals_info:
                offspring -= 1
                new_individuals[0] = self.__indivuduals_list[self.best_individual_id]


            i = 0
            while i < offspring:
                crossover_count += 1
                parent1 = self.pickOne(individuals_info, sum_fitness)
                parent2 = self.pickOne(individuals_info, sum_fitness)
                if individuals_info[f"{parent1}"] > individuals_info[f"{parent2}"]:
                    new_individuals[crossover_count] = self.__indivuduals_list[int(parent1)]
                elif individuals_info[f"{parent1}"] == individuals_info[f"{parent2}"]:
                    selected = random.randint(1, 2)
                    if selected == 1:
                        new_individuals[crossover_count] = self.__indivuduals_list[int(parent1)]
                    else:
                        new_individuals[crossover_count] = self.__indivuduals_list[int(parent2)]
                else:
                    new_individuals[crossover_count] = self.__indivuduals_list[int(parent2)]

                new_individuals[crossover_count].set_fitness(0)
                
                p1_connections = self.__indivuduals_list[int(parent1)].get_connection_list()
                c1_values = {f"{connection.get_ids()[0]}|{connection.get_ids()[1]}": connection.get_weight() for connection in p1_connections}
                p2_connections = self.__indivuduals_list[int(parent2)].get_connection_list()
                c2_values = {f"{connection.get_ids()[0]}|{connection.get_ids()[1]}": connection.get_weight() for connection in p2_connections}

                common = {connection: c1_values[connection] for connection in c1_values if connection in c2_values}

                new_individuals[crossover_count].set_connection_weight(common)

                i += 1

        # print(f"Generation: {self.generation_count}")
        self.__indivuduals_list = new_individuals
        self.generation_count += 1
    
    def mutate(self):
        for individual in self.__indivuduals_list:
            individual.mutate_connection_weights(self.__mutate_probs["connection_weight"])

            # add_connection_prob = random.uniform(0.0, 1.0)
            # if add_connection_prob <= self.__mutate_probs["add_connection"]:
                # individual.add_connection(self.__allow_recurrency)
            
            add_node_prob = random.uniform(0.0, 1.0)
            if add_node_prob <= self.__mutate_probs["add_node"]:
                individual.add_node()
            
            # individual.mutate_connection_state(self.__mutate_probs["connection_state"])

            # individual.mutate_node_state(self.__mutate_probs["node_state"])

    def get_best_individual_species(self) -> list[Brain]:
        pass

    def get_best_individual_population(self) -> Brain:
        pass

    def save_population(self):
        pass

    def load_population(self):
        pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.debug(f"Genome connections hashtable: {GENOME_HASHTABLE}")

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
- [ ] Threads
- [ ] Ajustar threshold e fitness
- [ ] Corrigir os bugs
- [ ] Testes de performance
- [ ] Bias
- [ ] Conexões recorrentes
"""