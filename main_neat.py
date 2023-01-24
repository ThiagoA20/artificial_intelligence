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

        n_radius = 20
        container_w = 60

        neuron_position = {}
        if self.__layers == {}:
            self.set_layers()
        layer_values = list(self.__layers.values())

        x_align = (c_width - (container_w * len(self.__layers))) / (len(self.__layers) + 1)
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
    
    def load_inputs(self, input_list: list[int]):
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
                input_list = []
                for connection in self.__connection_list:
                    connection_ids = connection.get_ids()
                    if connection_ids[1] == neuron:
                        input_list.append(self.__neuron_list[connection_ids[0]].get_output() * connection.get_weight())
                        # print(f"{connection_ids}: {self.__neuron_list[connection_ids[0]].get_output()}")
                self.__neuron_list[neuron].calculate_sum(input_list)
                self.__neuron_list[neuron].activate_neuron()
                # print(f"{neuron}: {self.__neuron_list[neuron].get_output()}")
    
    def get_outputs(self) -> list:
        output_values = []
        for i in range(self.__input_neurons+1, self.__input_neurons + self.__output_neurons + 1):
            output_values.append(self.__neuron_list[i].get_output())
        # print(output_values)
        return output_values


class Specie:
    def __init__(self):
        pass


class Population:
    def __init__(self):
        pass

# ----- Run pygame app ------------------------------------------------------------------------

def main(brain):
    global running, screen
    pygame.init()

    brain.load_inputs([99, 99, 92, 94, 95, 91, 95])
    brain_layers = brain.get_layers()
    logger.debug(f"Total layers: {len(brain_layers)} -> Layers: {brain_layers}")
    brain.run_network()

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        screen.fill((25, 25, 25))
        brain.draw_network()

        pygame.display.update()
    
    pygame.quit()

if __name__ == '__main__':
    running = True
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    my_brain = Brain(7, 1, 5, 100)
    logger.debug(f"Genome connections hashtable: {GENOME_HASHTABLE}")
    screen = pygame.display.set_mode([WIDTH, HEIGHT], RESIZABLE)
    main(my_brain)