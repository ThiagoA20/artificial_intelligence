import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import math
import random
import pygame
from pygame.locals import *
import logging
import time

logger = logging.getLogger(__name__)

# ----- Pygame Draw network --------------------------------------------------------------------

WIDTH = 1280
HEIGHT = 720    
ICON = pygame.image.load('D:\\projects\\enigma\\common\\graphical_interface\\brain_analyser\\assets\\ba_icon.png')
pygame.display.set_icon(ICON)
clock = pygame.time.Clock()
pygame.display.set_caption('Brain analyser')
running = False

INPUT_COLOR = (255, 25, 25)
HIDDEN_COLOR = (75, 75, 75)
OUTPUT_COLOR = (25, 255, 25)

pygame.font.init()
font = pygame.font.SysFont('rockwell', 17)
big_font = pygame.font.SysFont('rockwell', 21)

def newPosition(posy, posx, width1, height1, width2, height2):
    proportion_y = posy / height1
    proportion_x = posx / width1

    return (round(proportion_x * width2), round(proportion_y * height2))

class DrawItem:
    def __init__(self, posx, posy, width, height, color, target=None, center=True):
        if center:
            self.x = posx - width/2
            self.y = posy - height/2
        else:
            self.x = posx
            self.y = posy
        self.w = width
        self.h = height
        self.rect = pygame.Rect(posx, posy, width, height)
        self.color = color
        self.target = target

    def draw(self, item_posx, item_posy, width, height):
        pygame.draw.rect(screen, self.color, (item_posx, item_posy, width, height), 3)
    
    def check_event(self, posx, posy):
        pass

    def react_event(self):
        pass

class Label(DrawItem):
    def __init__(self, font, text, color, position, anchor="left"):
        self.image = font.render(text, 1, color)
        self.text = text
        self.color = color
        self.cursor = pygame.SYSTEM_CURSOR_HAND
        self.rect = self.image.get_rect()
        self.position = position
        self.anchor = anchor
        self.rect.top = position[1]
        self.rect.right = position[0]
        # setattr(self.rect, anchor, position)

    def draw(self):
        if HEIGHT > 900:
            self.image = big_font.render(self.text, 1, self.color)
            self.rect = self.image.get_rect()
        else:
            self.image = font.render(self.text, 1, self.color)
            self.rect = self.image.get_rect()
        np = newPosition(self.position[1], self.position[0], 1280, 720, WIDTH, HEIGHT)
        if self.anchor == "left":
            txt_pos = pygame.Rect(np[0], np[1], self.rect.w, self.rect.h)
        else:
            txt_pos = pygame.Rect(np[0], np[1], self.rect.w, self.rect.h)
            txt_pos.right = np[0]
        screen.blit(self.image, txt_pos)
    
    def getRect(self):
        return self.rect

def draw_neuron(position, type, radius):
    global screen
    if type == "HIDDEN":
        pygame.draw.circle(screen, HIDDEN_COLOR, position, radius)
    elif type == "OUTPUT":
        pygame.draw.circle(screen, OUTPUT_COLOR, position, radius, 2)
    else:
        pygame.draw.circle(screen, INPUT_COLOR, position, radius, 2)

def draw_connection(position1, position2):
    global screen
    pygame.draw.line(screen, (150, 150, 150), position1, position2)

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
    
    def activate_neuron(self):
        if (self.__neuron_type == 1):
            self.__sum_output = self.__sum_input
            self.__sum_input = 0
        else:
            self.__sum_output = self.__activation(self.__sum_input)
            self.__sum_input = 0


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
    
    def set_layers(self): 
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
        if self.__layers == {}:
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
        global screen, font
        start_x = 340
        start_y = 60
        c_width = 600
        c_height = 600

        """Make it proportional to the width and height of the network canvas"""
        n_radius = 20
        container_w = 60

        neuron_position = {}
        """Ajust set layers to always start with some content to avoid check all those if statements"""
        if self.__layers == {}:
            self.set_layers()
        layer_values = list(self.__layers.values())

        x_align = (c_width - (container_w * len(self.__layers))) / (len(self.__layers) + 1)
        """Set the option as a boolean to draw the canvas when needed for debug"""
        # pygame.draw.rect(screen, (255, 255, 0), (start_x, start_y, c_width, c_height), 3)
        for layer_num in range(len(self.__layers)):
            # pygame.draw.rect(screen, (255, 255, 255), (start_x + x_align * (layer_num + 1) + container_w * layer_num, start_y, container_w, c_height), 3)
            y_align = (c_height - (n_radius * len(layer_values[layer_num]))) / (len(layer_values[layer_num]) + 1)
            for neuron_num in range(len(layer_values[layer_num])):
                pos_x = start_x + container_w/2 + x_align * (layer_num + 1) + container_w * layer_num
                pos_y = start_y + n_radius/2 + y_align * (neuron_num + 1) + n_radius * neuron_num

                neuron_position[layer_values[layer_num][neuron_num]] = (pos_x, pos_y)

                if layer_num == 0:
                    neuron_type = "INPUT"
                elif layer_num == len(self.__layers) - 1:
                    neuron_type = "OUTPUT"
                else:
                    neuron_type = "HIDDEN"

                draw_neuron((pos_x, pos_y), neuron_type, n_radius)

                neuron_id = Label(font, f"{layer_values[layer_num][neuron_num]}", (255, 255, 255), (pos_x - n_radius/2.5, pos_y - n_radius/2))
                neuron_id.draw()
                neuron_opt = Label(font, f"{round(self.__neuron_list[layer_values[layer_num][neuron_num]].get_output(), 4)}", (255, 165, 0), (pos_x + n_radius, pos_y - n_radius * 2))
                neuron_opt.draw()

        for connection in self.__connection_list:
            connection_id = connection.get_ids()
            n1_pos = neuron_position[connection_id[0]]
            n2_pos = neuron_position[connection_id[1]]
            draw_connection((n1_pos[0] + n_radius, n1_pos[1]), (n2_pos[0] - n_radius, n2_pos[1]))

            connection_weight = Label(font, f"{round(connection.get_weight(), 3)}", (150, 150, 150), ((n1_pos[0] + n2_pos[0])/2, (n1_pos[1] + n2_pos[1])/2))
            connection_weight.draw()

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
                    if connection_ids[1] == neuron:
                        input_list.append(self.__neuron_list[connection_ids[0]].get_output() * connection.get_weight())
                        # print(f"{connection_ids}: {self.__neuron_list[connection_ids[0]].get_output()}")
                self.__neuron_list[neuron].calculate_sum(input_list)
                self.__neuron_list[neuron].activate_neuron()
                # print(f"{neuron}: {self.__neuron_list[neuron].get_output()}")
    
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

    def add_connection(self):
        global GENOME_HASHTABLE, INNOVATION_NUM
        """
        Dois neurônios na rede são escolhidos aleatoriamente
        
        Validação:
        - Não pode já haver uma conexão entre os neurônios
        - Não pode escolher duas vezes o mesmo neurônio
        - Não pode conectar dois neurônios na mesma camada
        - Não pode conectar um neurônio de uma camada maior com um neurônio em uma camada menor, a não ser que conexões recorrentes sejam permitidas
        - O peso para a nova conexão é definido como um valor aleatório
        - O innovation id é pego da tabela se já existir, se não, é adicionado mais um no contador e definido o valor na tabela
        - A conexão é definida como ativada
        - Como desativar uma conexão?
        
        Probabilidade:
        - Para o XOR, há 5% de chance de uma conexão ser adicionada depois do crossover
        - Permitir 20 tentativas de conseguir um par de nodos válidos para criar a conexão que não quebra as regras de validação
        - Se há uma conexão entre dois neurônios e está desativada, há 25% de chance de ela ser reativada
        """
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
            for camada in range(n1_layer, len(self.__layers)):
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
        
        # Criar conexões recorrentes

    def add_node(self):
        """
        Uma conexão foward ativada é escolhida aleatoriamente
        
        - A conexão é desativada
        - Um novo neurônio é colocado no array de neurônios
        - Duas novas conexões são adicionadas no array de conexões, uma conectada com a camada anterior e o novo neurônio e a outra conectada com o novo neurônio e a camada posterior
        - Uma das conexões recebe como valor para o peso o valor da conexão que foi desativada e a outra recebe um novo valor aleatório
        - As camadas são redefinidas
        - Se um neurônio surgir em um ponto de modo que faça uma conexão se tornar recorrente, então o valor de isRecurrent é definido como verdadeiro e a conexão é desativada, o contrário também é válido se uma conexão que antes era recorrente virar foward então é definido que ela não é recorrente mudando o valor de isRecurrent para falso
        """
        self.set_layers()
        selected_connection = self.__connection_list[random.randint(0, len(self.__connection_list) - 1)]
        selected_connection.change_state()
        new_neuron = Neuron()
        return selected_connection.get_info()
        # self.set_layers()
        # Depois de adicionar o neurônio, verificar cada camada para ver se foi
        # formada uma conexão recorrente, se sim, desativar a conexão
        # Verificar se existem conexões recorrentes desativadas e checkar se elas
        # ainda são recorrentes, se não, mudar o estado de recorrente para false

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

    def mutate_weights(self):
        pass
    

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
    def __init__(self, popsize: int, brain_settings: dict, allow_bias: bool, allow_recurrency: bool):
        self.__population_size = popsize
        self.__brain_settings = brain_settings
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
        
        # print(len(new_individuals))

        # print(f"Generation: {self.generation_count}")
        self.__indivuduals_list = new_individuals
        self.generation_count += 1
    
    def mutate(self):
        pass

    def get_best_individual_species(self) -> list[Brain]:
        pass

    def get_best_individual_population(self) -> Brain:
        pass

    def save_population(self):
        pass

    def load_population(self):
        pass

# ----- Run pygame app ------------------------------------------------------------------------

def my_fitness(output_list: list, answers: list) -> float:
    "output list and answers must be the same size, and the output of this function must be positive"
    fitness = 0
    for i in range(len(output_list)):
        if answers[i] >= 0.5:
            answer = 1
        else:
            answer = 0
        fitness += 1 - abs(output_list[i] - answer)
    return fitness

inputs_and_answers = {
    "IP1": [[1, 1], [0]],
    "IP2": [[1, 0], [1]],
    "IP3": [[0, 1], [1]],
    "IP4": [[0, 0], [0]]
}

def main(population):
    global running, screen
    pygame.init()
    
    for i in range(100):
        for input_value in inputs_and_answers:
            population.set_inputs(inputs_and_answers[input_value][0])
            population.run_simulation()
            population.calculate_fitness(my_fitness, inputs_and_answers[input_value][1])
        population.speciation()
        population.crossover()
        # population.mutate()
    # brain.load_inputs([99, 99, 92, 94, 95, 91, 95])
    # brain_layers = brain.get_layers()
    # logger.debug(f"Total layers: {len(brain_layers)} -> Layers: {brain_layers}")
    # brain.run_network()
    # brain.add_connection()
    # max_fit = 0
    # for individual_fitness in population.get_fitness():
    #     fit_value = list(individual_fitness.values())[0]
    #     if fit_value > max_fit:
    #         max_fit = fit_value
        # print(individual_fitness)
    # print(max_fit)

    # for specie in population.get_species():
        # print(specie)

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        screen.fill((25, 25, 25))
        population.draw_fittest_network()

        pygame.display.update()
    
    pygame.quit()

if __name__ == '__main__':
    running = True
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    my_population = Population(50, {"INPUTS": 2, "HIDDEN": 0, "OUTPUTS": 1, "CONNECTIONS": 100}, False, False)

    logger.debug(f"Genome connections hashtable: {GENOME_HASHTABLE}")
    screen = pygame.display.set_mode([WIDTH, HEIGHT], RESIZABLE)
    main(my_population)

"""
O id dos neurônios não importa, por que em esscência, o que estamos buscando é
a informação da melhor composição da rede e essa informação já está nas conexões,
um neurônio pode mudar de posição, mas o que importa é saber se deve haver um novo
neurônio em dada posição e quais conexões esse neurônio deve ter.

Se pensarmos em uma rede neural como uma estrutura que correlaciona fenômenos de causa e efeito no universo, podemos inferir que achariamos a velocidade da luz colocando o valor de energia nos inputs e massa nos outputs, então toda aquela abstração no meio poderia ser resumida a um número que é a velocidade da luz ao quadrado, a seleção natural iria eliminar as entradas irrelevantes se começarmos com todos os neurônios desconectados e sem neurônios escondidos.
---

- [X] Especiação
- [ ] Crossover
- [ ] Mutação
- [ ] Threads
- [ ] Testes de performance
- [ ] Bias
- [ ] Conexões recorrentes
"""