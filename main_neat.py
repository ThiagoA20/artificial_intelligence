import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import math
import random
import pygame
from pygame.locals import *
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

WIDTH = 1280
HEIGHT = 720    
ICON = pygame.image.load('D:\\projects\\enigma\\common\\graphical_interface\\brain_analyser\\assets\\ba_icon.png')
pygame.display.set_icon(ICON)
clock = pygame.time.Clock()
pygame.display.set_caption('Brain analyser')
running = False


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

INPUT_COLOR = (255, 25, 25)
HIDDEN_COLOR = (75, 75, 75)
OUTPUT_COLOR = (25, 255, 25)

POPSIZE = 0
INPUT_NEURONS = 2
HIDDEN_NEURONS = 0
OUTPUT_NEURONS = 1
CONNECTIONS_PERCENTAGE = 25
DEFAULT_ACTIVATION = sigmoid

INNOVATION_NUM = 0

GENOME_HASHTABLE = {}

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
    
    def set_layer(self, layer):
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
        else:
            self.__sum_output = self.__activation(self.__sum_input)


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


class Specie:
    def __init__(self):
        pass


class Brain:
    def __init__(self, input_neurons: int = INPUT_NEURONS, hidden_neurons: int = HIDDEN_NEURONS, output_neurons: int = OUTPUT_NEURONS, connections_percentage: int = CONNECTIONS_PERCENTAGE):
        """Set the innovation number"""
        logger.debug(f"NN info: ipn = {input_neurons}; hn = {hidden_neurons}; opn = {output_neurons}; ic% = {connections_percentage}")
        global GENOME_HASHTABLE
        global INNOVATION_NUM
        self.__neuron_list = []
        neuron_counter = 0
        total_neurons = input_neurons + output_neurons + hidden_neurons + 1
        while neuron_counter < total_neurons:
            if neuron_counter == 0:
                self.__neuron_list.append(Neuron(neuron_counter, 3, 0))
            else:
                if neuron_counter <= input_neurons:
                    self.__neuron_list.append(Neuron(neuron_counter, 1, 0))
                elif neuron_counter <= input_neurons + output_neurons:
                    self.__neuron_list.append(Neuron(neuron_counter, 2, 2))
                else:
                    self.__neuron_list.append(Neuron(neuron_counter, 0, 1))
            neuron_counter += 1
        self.__connection_list = []
        real_connections = lambda x: math.ceil(x * connections_percentage/100)
        if hidden_neurons > 0:
            connection_amount = (input_neurons * hidden_neurons) + (hidden_neurons * output_neurons)
        else:
            connection_amount = (input_neurons * output_neurons)
        total_connections = real_connections(connection_amount)
        while total_connections > 0:
            logger.debug(f"Total connections: {total_connections}")
            if hidden_neurons == 0:
                for opn in range(1, output_neurons + 1):
                    for ipn in range(1, input_neurons + 1):
                        if f"{ipn}|{input_neurons + opn}" in GENOME_HASHTABLE:
                            self.__connection_list.append(Connection(GENOME_HASHTABLE[f"{ipn}|{input_neurons + opn}"], ipn, input_neurons + opn, random.uniform(-20, 20)))
                            total_connections -= 1
                        else:
                            INNOVATION_NUM += 1
                            GENOME_HASHTABLE[f"{ipn}|{input_neurons + opn}"] = INNOVATION_NUM
                            self.__connection_list.append(Connection(GENOME_HASHTABLE[f"{ipn}|{input_neurons + opn}"], ipn, input_neurons + opn, random.uniform(-20, 20)))
                            total_connections -= 1
                        if total_connections == 0:
                            break
                    if total_connections == 0:
                            break
            else:
                for hn in range(1, hidden_neurons + 1):
                    for ipn in range(1, input_neurons + 1):
                        if f"{ipn}|{input_neurons + hn}" in GENOME_HASHTABLE:
                            self.__connection_list.append(Connection(GENOME_HASHTABLE[f"{ipn}|{input_neurons + hn}"], ipn, input_neurons + hn, random.uniform(-20, 20)))
                            total_connections -= 1
                        else:
                            INNOVATION_NUM += 1
                            GENOME_HASHTABLE[f"{ipn}|{input_neurons + hn}"] = INNOVATION_NUM
                            self.__connection_list.append(Connection(GENOME_HASHTABLE[f"{ipn}|{input_neurons + hn}"], ipn, input_neurons + hn, random.uniform(-20, 20)))
                            total_connections -= 1
                        if total_connections == 0:
                            break
                    for opn in range(1, output_neurons + 1):
                        if f"{hn}|{input_neurons + hidden_neurons + opn}" in GENOME_HASHTABLE:
                            self.__connection_list.append(Connection(GENOME_HASHTABLE[f"{hn}|{input_neurons + hidden_neurons + opn}"], hn, input_neurons + hidden_neurons + opn, random.uniform(-20, 20)))
                            total_connections -= 1
                        else:
                            INNOVATION_NUM += 1
                            GENOME_HASHTABLE[f"{hn}|{input_neurons + hidden_neurons + opn}"] = INNOVATION_NUM
                            self.__connection_list.append(Connection(GENOME_HASHTABLE[f"{hn}|{input_neurons + hidden_neurons + opn}"], hn, input_neurons + hidden_neurons + opn, random.uniform(-20, 20)))
                            total_connections -= 1
                        if total_connections == 0:
                            break
                    if total_connections == 0:
                        break
        neuron_numbers = [(neuron.get_id(), neuron.get_type()) for neuron in self.__neuron_list]
        logger.debug(f"Neuron numbers/type: {neuron_numbers}")
    
    def set_layers(self): 
        layers = {}
        input_list = []
        output_list = []
        for connection in self.__connection_list:
            neuron_ids = connection.get_ids()
            input_list.append(neuron_ids[0])
            output_list.append(neuron_ids[1])
    
    def draw_network(self):
        pass

    def get_neuron_list(self) -> list[Neuron]:
        return self.__neuron_list
    
    def get_connection_list(self) -> list[Connection]:
        return self.__connection_list

def draw_neuron(position, type):
    global screen
    if type == "HIDDEN":
        pygame.draw.circle(screen, HIDDEN_COLOR, position, 20)
    elif type == "OUTPUT":
        pygame.draw.circle(screen, OUTPUT_COLOR, position, 20, 2)
    else:
        pygame.draw.circle(screen, INPUT_COLOR, position, 20, 2)

def draw_connection(position1, position2):
    global screen
    pygame.draw.line(screen, (150, 150, 150), position1, position2)

Neurons = {
    "INPUT": [(500, 180), (500, 280), (500, 380), (500, 480)], 
    "HIDDEN": [(600, 230), (600, 330), (600, 430), (700, 230), (700, 330), (700, 430)],
    "OUTPUT": [(800, 280),(800, 380)]
}

Connections = [
    [(520, 180), (580, 230)],
    [(520, 180), (580, 330)],
    [(520, 180), (580, 430)],
    [(520, 280), (580, 230)],
    [(520, 280), (580, 330)],
    [(520, 280), (580, 430)],
    [(520, 380), (580, 230)],
    [(520, 380), (580, 330)],
    [(520, 380), (580, 430)],
    [(520, 480), (580, 230)],
    [(520, 480), (580, 330)],
    [(520, 480), (580, 430)],
    [(620, 230), (680, 230)],
    [(620, 330), (680, 330)],
    [(620, 430), (680, 430)],
    [(720, 230), (780, 280)],
    [(720, 230), (780, 380)],
    [(720, 330), (780, 280)],
    [(720, 330), (780, 380)],
    [(720, 430), (780, 280)],
    [(720, 430), (780, 380)]
]


def main(brain):
    global running, screen
    pygame.init()

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        screen.fill((25, 25, 25))
        for key in Neurons:
            for position in Neurons[key]:
                draw_neuron(position, key)
        for connection in Connections:
            draw_connection(connection[0], connection[1])

        pygame.display.update()
    
    pygame.quit()

if __name__ == '__main__':
    running = True
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(message)s")
    my_brain = Brain(7, 0, 5, 100)
    logger.debug(f"Genome connections hashtable: {GENOME_HASHTABLE}")
    screen = pygame.display.set_mode([WIDTH, HEIGHT], RESIZABLE)
    main(my_brain)