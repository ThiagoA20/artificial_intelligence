import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
import global_vars

WIDTH = 1280
HEIGHT = 720    
ICON = pygame.image.load('D:\\projects\\enigma\\common\\graphical_interface\\brain_analyser\\assets\\ba_icon.png')
pygame.display.set_icon(ICON)
clock = pygame.time.Clock()
pygame.display.set_caption('Brain analyser')

INPUT_COLOR = (255, 25, 25)
HIDDEN_COLOR = (75, 75, 75)
OUTPUT_COLOR = (25, 255, 25)
GRAY_BACKGROUND = (25, 25, 25)

pygame.font.init()
font = pygame.font.SysFont('rockwell', 17)
big_font = pygame.font.SysFont('rockwell', 21)
screen = None

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


def draw_connection(position1, position2, color=""):
    global screen
    if color == "red":
        pygame.draw.line(screen, (255, 0, 0), position1, position2)
    else:
        pygame.draw.line(screen, (150, 150, 150), position1, position2)


def draw_network(layers: dict, connections: dict, neurons: dict, draw_canvas=False, show_disabled=False):
    global screen, font
    start_x = 340
    start_y = 60
    c_width = 600
    c_height = 600

    """Make it proportional to the width and height of the network canvas"""
    n_radius = 20
    container_w = 60

    layer_values = list(layers.values())
    x_align = (c_width - (container_w * len(layers))) / (len(layers) + 1)

    if draw_canvas:
        pygame.draw.rect(screen, (255, 255, 0), (start_x, start_y, c_width, c_height), 3)
    
    # Calculate the position of every neuron
    for layer_num in range(len(layers)):
        if draw_canvas:
            pygame.draw.rect(screen, (255, 255, 255), (start_x + x_align * (layer_num + 1) + container_w * layer_num, start_y, container_w, c_height), 3)
        y_align = (c_height - (n_radius * len(layer_values[layer_num]))) / (len(layer_values[layer_num]) + 1)
        for neuron_num in range(len(layer_values[layer_num])):
            pos_x = start_x + container_w/2 + x_align * (layer_num + 1) + container_w * layer_num
            pos_y = start_y + n_radius/2 + y_align * (neuron_num + 1) + n_radius * neuron_num
            neurons[str(layer_values[layer_num][neuron_num])].append((pos_x, pos_y))
    
    # Draw connections and connection weights
    for connection in connections:
        connection_ids = tuple(connection.split('|'))
        n1_pos = neurons[connection_ids[0]][1]
        n2_pos = neurons[connection_ids[1]][1]
        if show_disabled:
            if connections[connection][0]:
                color = "gray"
                connection_weight = Label(font, f"{round(connections[connection][1], 3)}", (150, 150, 150), ((n1_pos[0] + n2_pos[0])/2, (n1_pos[1] + n2_pos[1])/2))
                connection_weight.draw()
            else:
                color = "red"
            draw_connection((n1_pos[0] + n_radius, n1_pos[1]), (n2_pos[0] - n_radius, n2_pos[1]), color)
        else:
            if connections[connection][0]:
                connection_weight = Label(font, f"{round(connections[connection][1], 3)}", (150, 150, 150), ((n1_pos[0] + n2_pos[0])/2, (n1_pos[1] + n2_pos[1])/2))
                connection_weight.draw()
                draw_connection((n1_pos[0] + n_radius, n1_pos[1]), (n2_pos[0] - n_radius, n2_pos[1]))
    
    # Draw neurons, neuron ids and output
    for neuron in neurons:
        neuron_pos = neurons[neuron][1]
        if int(neuron) in layer_values[0]:
            neuron_color = "INPUT"
        elif int(neuron) in layer_values[-1]:
            neuron_color = "OUTPUT"
        else:
            neuron_color = "HIDDEN"
        draw_neuron(neuron_pos, neuron_color, n_radius)
        neuron_id = Label(font, f"{neuron}", (255, 255, 255), (neuron_pos[0] - n_radius/2.5, neuron_pos[1] - n_radius/2))
        neuron_id.draw()
        neuron_opt = Label(font, f"{round(neurons[neuron][0], 4)}", (255, 165, 0), (neuron_pos[0] + n_radius, neuron_pos[1] - n_radius * 2))
        neuron_opt.draw()


def draw_network_info(parameters: dict):
    individuals_label = Label(font, f"Individuals: {parameters['individuals']}", (150, 150, 150), (30, 30))
    species_label = Label(font, f"Species: {parameters['species']}", (150, 150, 150), (30, 60))
    generation_label = Label(font, f"Generation: {parameters['generation']}", (150, 150, 150), (30, 90))
    best_individual_label = Label(font, f"Best individual: {parameters['best_individual']}", (150, 150, 150), (30, 120))
    fitness_label = Label(font, f"Best Fitness: {parameters['best_fitness']}", (150, 150, 150), (30, 150))
    connection_weight_label = Label(font, f"Connection weight prob: {parameters['connection_weight']}%", (150, 150, 150), (30, 390))
    add_connection_label = Label(font, f"Add connection prob: {parameters['add_connection']}%", (150, 150, 150), (30, 420))
    add_node_label = Label(font, f"Add node prob: {parameters['add_node']}%", (150, 150, 150), (30, 450))
    connection_state_label = Label(font, f"Connection state prob: {parameters['connection_state']}%", (150, 150, 150), (30, 480))
    node_state_label = Label(font, f"Node state prob: {parameters['node_state']}%", (150, 150, 150), (30, 510))
    allow_bias_label = Label(font, f"Allow bias: {parameters['allow_bias']}", (150, 150, 150), (30, 540))
    allow_recurrency_label = Label(font, f"Allow recurrency: {parameters['allow_recurrency']}", (150, 150, 150), (30, 570))
    input_neurons_label = Label(font, f"Start input neurons: {parameters['input_neurons']}", (150, 150, 150), (30, 600))
    hidden_neurons_label = Label(font, f"Start hidden neurons: {parameters['hidden_neurons']}", (150, 150, 150), (30, 630))
    output_neurons_label = Label(font, f"Start output neurons: {parameters['output_neurons']}", (150, 150, 150), (30, 660))

    parameter_list = [
        individuals_label,
        species_label,
        generation_label,
        best_individual_label,
        fitness_label,
        connection_weight_label,
        add_connection_label,
        add_node_label,
        connection_state_label,
        node_state_label,
        allow_bias_label,
        allow_recurrency_label,
        input_neurons_label,
        hidden_neurons_label,
        output_neurons_label
    ]

    for parameter in parameter_list:
        parameter.draw()

# def draw_data_info():
    # pass

def brain_analyser():
    global screen
    screen = pygame.display.set_mode([WIDTH, HEIGHT], RESIZABLE)
    pygame.init()

    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                with global_vars.running_lock:
                    global_vars.running = False
        screen.fill(GRAY_BACKGROUND)
        
        with global_vars.network_lock:
            network = global_vars.network

        draw_network(
            layers=network[0],
            connections=network[1],
            neurons=network[2]
        )

        with global_vars.network_info_lock:
            network_info = global_vars.network_info

        draw_network_info(
            parameters=network_info
        )

        # draw_data_info()

        pygame.display.update()
        with global_vars.running_lock:
            if not global_vars.running:
                break
    
    pygame.quit()
