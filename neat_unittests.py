import unittest
from main_neat import *
import os
from math import ceil

def my_fitness(output_list: list, answers: list) -> float:
    "output list and answers must be the same size, and the output of this function must be positive"
    error = round(abs(answers[0] - output_list[0]), 6)
    fitness = round(1 - error, 6)
    return fitness

class ActivationTestCase(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(sigmoid(1), 0.731059)
        self.assertEqual(sigmoid(-31), 0.0)
        self.assertEqual(sigmoid(0.731059), 0.675038)
        self.assertEqual(sigmoid(-3132.15641), 0)
        self.assertEqual(sigmoid(1234010234), 1.0)
        self.assertEqual(sigmoid(12), 0.999994)
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(3.435), 0.968781)

    def test_ReLU(self):
        self.assertEqual(ReLU(0.731059), 0.731059)
        self.assertEqual(ReLU(42), 42)
        self.assertEqual(ReLU(-15), 0.0)

    def test_Step(self):
        self.assertEqual(Step(-52), 0.0)
        self.assertEqual(Step(0), 0.0)
        self.assertEqual(Step(0.3455), 1.0)


class NeuronTestCase(unittest.TestCase):

    def test_initialize(self):
        # 0: HIDDEN, 1: SENSOR, 2: OUTPUT, 3: BIAS
        new_neuron = Neuron(neuron_id=1, neuron_type=1, neuron_layer=1)
        self.assertEqual(new_neuron.get_type(), 1)
        self.assertEqual(new_neuron.get_id(), 1)
        self.assertEqual(new_neuron.get_layer(), 1)
        new_neuron.set_layer(4)
        self.assertEqual(new_neuron.get_layer(), 4)
        new_neuron.change_state()
        self.assertEqual(new_neuron.get_neuron_info(), {"id": 1, "type": 1, "layer": 4, "Sum result": 0, "output": 0, "Active": False})
    
    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            Neuron(neuron_id=None, neuron_type=1, neuron_layer=1)
        with self.assertRaises(TypeError):
            Neuron(neuron_id=1, neuron_type=None, neuron_layer=1)
        with self.assertRaises(TypeError):
            Neuron(neuron_id=1, neuron_type=1, neuron_layer=None)
    
    def test_invalid_ranges(self):
        with self.assertRaises(ValueError):
            Neuron(neuron_id=1, neuron_type=-2, neuron_layer=1)
        with self.assertRaises(ValueError):
            Neuron(neuron_id=1, neuron_type=999, neuron_layer=1)
    
    def test_calculate_sum(self):
        new_neuron = Neuron(neuron_id=1, neuron_type=1, neuron_layer=1)
        new_neuron.calculate_sum([20, 10, 35, 12, 42])
        self.assertEqual(new_neuron.get_neuron_info()["Sum result"], 119)
        with self.assertRaises(TypeError):
            new_neuron.calculate_sum(None)
        self.assertEqual(new_neuron.get_sum(), 119)
    
    def test_calculate_sum2(self):
        new_neuron = Neuron(neuron_id=1, neuron_type=1, neuron_layer=1)
        new_neuron.calculate_sum([-2.0, 5.4, 3.25, 1.2, -4.2])
        self.assertEqual(new_neuron.get_sum(), 3.65)
    
    def test_calculate_activation_function(self):
        new_neuron = Neuron(neuron_id=1, neuron_type=0, neuron_layer=1)
        new_neuron.calculate_sum([-2.0, 5.4, 3.25, 1.2, -4.2])
        new_neuron.activate_neuron()
        self.assertEqual(new_neuron.get_output(), 0.974667)
    
    def test_calculate_activation_function2(self):
        new_neuron = Neuron(neuron_id=1, neuron_type=1, neuron_layer=1)
        new_neuron.calculate_sum([-2.0, 5.4, 3.25, 1.2, -4.2])
        new_neuron.activate_neuron()
        self.assertEqual(new_neuron.get_output(), 3.65)


class ConnectionTestCase(unittest.TestCase):

    def test_initialize(self):
        new_connection = Connection(in_neuron_id=2, out_neuron_id=5, innovation_number=1)
        self.assertEqual(new_connection.get_info(), {"innovation number": 1, "in neuron": 2, "out neuron": 5, "weight": 0.0, "active": True, "recurrent": False})

    def test_wrong_neuron_id(self):
        with self.assertRaises(ValueError):
            new_connection = Connection(in_neuron_id=2, out_neuron_id=2, innovation_number=1)
    
    def test_change_connection_weight(self):
        new_connection = Connection(in_neuron_id=2, out_neuron_id=5, innovation_number=1)
        self.assertEqual(new_connection.get_weight(), 0)
        new_connection.set_weight(weight=1.348)
        self.assertEqual(new_connection.get_weight(), 1.348)
    
    def test_activate_and_deactivate_connection(self):
        new_connection = Connection(in_neuron_id=2, out_neuron_id=5, innovation_number=1)
        self.assertTrue(new_connection.is_active())
        new_connection.change_state()
        self.assertFalse(new_connection.is_active())
    
    def test_change_recurrency(self):
        new_connection = Connection(in_neuron_id=1, out_neuron_id=3, innovation_number=5)
        self.assertFalse(new_connection.is_recurrent())
        new_connection.change_recurrency()
        self.assertTrue(new_connection.is_recurrent())
    
    def test_check_data_integrity(self):
        with self.assertRaises(TypeError):
            Connection(in_neuron_id=None, out_neuron_id=5, innovation_number=1)
        with self.assertRaises(TypeError):
            Connection(in_neuron_id=2, out_neuron_id=5, innovation_number=None)
        with self.assertRaises(TypeError):
            Connection(in_neuron_id=2, out_neuron_id=None, innovation_number=1)
        with self.assertRaises(TypeError):
            Connection(in_neuron_id=2, out_neuron_id=5, innovation_number=1, weight=None)
        with self.assertRaises(TypeError):
            Connection(in_neuron_id=2, out_neuron_id=5, innovation_number=1, active=None)
        with self.assertRaises(TypeError):
            new_connection = Connection(in_neuron_id=2, out_neuron_id=5, innovation_number=1)
            new_connection.set_weight(weight=None)


class BrainTestCase(unittest.TestCase):

    def setUp(self):
        self.INPUT_NEURONS = 7
        self.HIDDEN_NEURONS = 1
        self.OUTPUT_NEURONS = 5
        self.CONNECTIONS_PERCENTAGE = 100
        self.new_brain = Brain(input_neurons=self.INPUT_NEURONS, hidden_neurons=self.HIDDEN_NEURONS, output_neurons=self.OUTPUT_NEURONS, connections_percentage=self.CONNECTIONS_PERCENTAGE)
        self.new_brain2 = Brain(3, 0, 2, 0)
        self.neuron_list = [
            Neuron(0, 3, 1),
            Neuron(1, 1, 1),
            Neuron(2, 1, 1),
            Neuron(3, 1, 1),
            Neuron(4, 2, 4),
            Neuron(5, 2, 4),
            Neuron(6, 0, 2),
            Neuron(7, 0, 2),
            Neuron(8, 0, 2),
            Neuron(9, 0, 2),
            Neuron(10, 0, 3),
            Neuron(11, 0, 3)
        ]
        self.connection_list = [
            Connection(1, 1, 6, 3.52),
            Connection(2, 1, 7, 1.25),
            Connection(3, 2, 7, 1.54),
            Connection(4, 2, 8, 2.21),
            Connection(5, 3, 8, 4.42),
            Connection(6, 3, 9, 1.37),
            Connection(7, 6, 10, 1.22),
            Connection(8, 6, 11, 2.27),
            Connection(9, 7, 10, 3.22),
            Connection(10, 7, 11, 4.11),
            Connection(11, 8, 10, 0.82),
            Connection(12, 8, 11, 1.15),
            Connection(13, 9, 10, 1.59),
            Connection(14, 9, 11, 3.77),
            Connection(15, 10, 4, 0.35),
            Connection(16, 11, 5, 0.67),
            Connection(17, 10, 4, 2.14),
            Connection(18, 11, 5, 1.97)
        ]

    def test_initialize(self):
        self.assertEqual(len(self.new_brain.get_neuron_list()), self.INPUT_NEURONS + self.HIDDEN_NEURONS + self.OUTPUT_NEURONS + 1)
        real_connections = lambda x: ceil(x * self.CONNECTIONS_PERCENTAGE / 100)
        if self.HIDDEN_NEURONS > 0:
            connection_amount = (self.INPUT_NEURONS * self.HIDDEN_NEURONS) + (self.HIDDEN_NEURONS * self.OUTPUT_NEURONS)
        else:
            connection_amount = (self.INPUT_NEURONS * self.OUTPUT_NEURONS)
        self.assertEqual(len(self.new_brain.get_connection_list()), real_connections(connection_amount))
    
    def test_initial_connection_list(self):
        """Quando tem apenas um neurônio escondido, conecta todos os neurônios de entrada a ele e depois conecta o neurônio escondido a todos os neurônios de saída"""
        total_connections = calculate_initial_connections(7, 1, 5, 100)
        connection_array = [connection.get_ids() for connection in generate_initial_connection_list(total_connections, 7, 1, 5)]
        self.assertEqual(connection_array, [(1, 13), (2, 13), (3, 13), (4, 13), (5, 13), (6, 13), (7, 13), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12)])
    
    def test_initial_connection_list2(self):
        """Quando não tem neurônios escondidos ele conecta o primeiro neurônio de entrada com todos de saída, depois repete passando para o segundo neurônio de entrada e assim sucessivamente"""
        total_connections = calculate_initial_connections(7, 0, 5, 100)
        connection_array = [connection.get_ids() for connection in generate_initial_connection_list(total_connections, 7, 0, 5)]
        self.assertEqual(connection_array, [(1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12)])
    
    def test_initial_connection_list3(self):
        """Quando tem mais de um neurônio escondido ele pega
        os neurônios de entrada, conecta com o primeiro neurônio escondido, depois pega esse neurônio escondido e conecta todos os neurônios de saída, depois repete, indo para o 2 neurônio escondido"""
        total_connections = calculate_initial_connections(2, 2, 1, 100)
        connection_array = [connection.get_ids() for connection in generate_initial_connection_list(total_connections, 2, 2, 1)]
        self.assertEqual(connection_array, [(1, 4), (2, 4), (4, 3), (1, 5), (2, 5), (5, 3)])
    
    def test_layers_7_1_5(self):
        test_brain = Brain(7, 1, 5, 100)
        test_brain.set_layers()
        self.assertEqual(test_brain.get_layers(), {"1": [1, 2, 3, 4, 5, 6, 7], "2": [13], "3": [8, 9, 10, 11, 12]})
    
    def test_layers_7_5(self):
        test_brain = Brain(7, 0, 5, 100)
        test_brain.set_layers()
        self.assertEqual(test_brain.get_layers(), {"1": [1, 2, 3, 4, 5, 6, 7], "2": [8, 9, 10, 11, 12]})
    
    def test_layers_2_2_1(self):
        test_brain = Brain(2, 2, 1, 100)
        test_brain.set_layers()
        self.assertEqual(test_brain.get_layers(), {"1": [1, 2], "2": [4, 5], "3": [3]})
    
    def test_layers_3_4_2_2(self):
        self.new_brain2.load_neuron_list(self.neuron_list)
        self.new_brain2.load_connection_list(self.connection_list)
        self.assertEqual(self.new_brain2.get_brain_info(), {"input neurons": 3, "hidden neurons": 0, "output neurons": 2, "connection percentage": 0, "neuron list": self.neuron_list, "connection list": self.connection_list, "layers": {}, "fitness": 0.0, "specie": None, "ajusted fitness": None})
        self.new_brain2.set_layers()
        self.assertEqual(self.new_brain2.get_layers(), {'1': [1, 2, 3], '2': [6, 7, 8, 9], '3': [10, 11], '4': [4, 5]})

    def test_load_inputs(self):
        with self.assertRaises(ValueError):
            self.new_brain.load_inputs([12, 32, 21])
        input_list = [10, 12, 32, 14, 85, 11, 75]
        self.new_brain.load_inputs(input_list)
        test_inputs = []
        for neuron in range(1, self.INPUT_NEURONS + 1):
            test_inputs.append(self.new_brain.get_neuron_list()[neuron].get_output())
        self.assertEqual(test_inputs, input_list)

    def test_run_network(self):
        new_brain = Brain(input_neurons=self.INPUT_NEURONS, hidden_neurons=self.HIDDEN_NEURONS, output_neurons=self.OUTPUT_NEURONS, connections_percentage=self.CONNECTIONS_PERCENTAGE)
        self.assertEqual(new_brain.get_outputs(), [0, 0, 0, 0, 0])
        input_list = [99, 99, 92, 94, 95, 91, 95]
        new_brain.load_inputs(input_list)
        new_brain.run_network()
        self.assertNotEqual(new_brain.get_outputs(), [0, 0, 0, 0, 0])
    
    def test_set_fitness(self):
        new_brain = Brain(2, 3, 1, 0)
        neuron_list = [
            Neuron(0, 3, 1),
            Neuron(1, 1, 1), # 0
            Neuron(2, 1, 1), # 0 ######################### 1.37 ######## 1.505 ####### 0.56
            Neuron(3, 2, 3), # (n4, n5, n6) --> sigmoid((0.5 * 2.74) + (0.5 * 3.01) + (0.5 * 1.12)) = sigmoid(3.435) = 0.968781
            Neuron(4, 0, 2), # (n1, n2) --> sigmoid((0 * 0.98) + (0 * 0.37)) = 0.5
            Neuron(5, 0, 2), # (n1, n2) --> sigmoid((0 * 1.22) + (0 * 0.64)) = 0.5
            Neuron(6, 0, 2), # (n1, n2) --> sigmoid((0 * 1.11) + (0 * 1.42)) = 0.5
        ]
        new_brain.load_neuron_list(neuron_list)
        connection_list = [
            Connection(1, 1, 4, 0.98),
            Connection(2, 1, 5, 1.22),
            Connection(3, 1, 6, 1.11),
            Connection(4, 2, 4, 0.37),
            Connection(5, 2, 5, 0.64),
            Connection(6, 2, 6, 1.42),
            Connection(7, 4, 3, 2.74),
            Connection(8, 5, 3, 3.01),
            Connection(9, 6, 3, 1.12),
        ]
        new_brain.load_connection_list(connection_list)
        inputs_and_answers = {
            "IP1": [[0, 0], [0], 0.968781, 0.968781, 0.031219],
            "IP2": [[1, 0], [1], 0.994283, 0.005717, 0.994283],
            "IP3": [[0, 1], [1], 0.988941, 0.011059, 0.988941],
            "IP4": [[1, 1], [0], 0.997035, 0.997035, 0.002965]
        }
        output_total = 0
        for value in inputs_and_answers:
            new_brain.load_inputs(inputs_and_answers[value][0])
            new_brain.run_network()
            output = new_brain.get_outputs()[0]
            output_total += output
            fitness = my_fitness([output], inputs_and_answers[value][1])
            new_brain.set_fitness(fitness)
            self.assertEqual(fitness, inputs_and_answers[value][4])
            self.assertEqual(output, inputs_and_answers[value][2])
        self.assertEqual(output_total, 3.94904)
        self.assertEqual(new_brain.get_fitness(), 2.017408)

    def test_reset_fitness(self):
        new_brain = Brain(2, 3, 1, 0)
        new_brain.set_fitness(1.25)
        self.assertEqual(new_brain.get_fitness(), 1.25)
        new_brain.reset_fitness()
        self.assertEqual(new_brain.get_fitness(), 0.0)

    # def test_add_connection(self):
    #     self.new_brain.add_connection()
    #     self.assertEqual(len(self.new_brain.get_connection_list()), 13)


class PopulationTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.brain_settings={
            "INPUTS": 2,
            "HIDDEN": 0,
            "OUTPUTS": 1,
            "CONNECTIONS": 100
        }
        self.new_population = Population(10, self.brain_settings, {}, False, False)
        self.individuals_list = [
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 2.134), Connection(2, 2, 3, 1.145)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.122), Connection(2, 2, 3, 1.197)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 1.314), Connection(2, 2, 3, 3.228)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 2.595), Connection(2, 2, 3, 0.733)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 3.157), Connection(2, 2, 3, 2.665)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.044), Connection(2, 2, 3, 2.227)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.198), Connection(2, 2, 3, 1.742)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.742), Connection(2, 2, 3, 0.178)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 1.967), Connection(2, 2, 3, 3.982)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.435), Connection(2, 2, 3, 0.994)])
        ]
        self.new_population.load_population(self.individuals_list, 0, 0.0, 0.0, 0)
        self.inputs_and_answers = {
            "IP1": [[0, 0], [0]],
            "IP2": [[1, 0], [1]],
            "IP3": [[0, 1], [1]],
            "IP4": [[1, 1], [0]]
        }

    def test_load_population(self):
        self.assertEqual(self.new_population.get_info(), {
            "popsize": 10, "brain_settings": self.brain_settings, "mutate_probs": {}, "allow_bias": False, "allow_recurrency": False, "individuals_list": self.individuals_list, "specie_list": [], "generation_count": 0, "threshold": 100.0, "species_target": 5, "pop_fitness": 0, "max_fitness": 0, "best_individual_id": 0, "threshold_change_ratio": 0.5
        })

    def test_save_population(self):
        self.new_population.save_population('test')
        self.assertTrue(os.path.exists('test.json'))
        os.remove('test.json')

    def test_initialize_from_file(self):
        if os.path.exists('test.json'):
            self.new_population.load_from_file('test')
            self.assertNotEqual(self.new_population.get_info(), {})
            os.remove('test.json')
        else:
            self.new_population.save_population('test')
            self.new_population.load_from_file('test')
            self.assertNotEqual(self.new_population.get_info(), {})
            os.remove('test.json')

    def test_calculate_fitness(self):
        for value in self.inputs_and_answers:
            self.new_population.set_inputs(self.inputs_and_answers[value][0])
            self.new_population.run_simulation()
            self.new_population.calculate_fitness(my_fitness, self.inputs_and_answers[value][1])
        self.new_population.save_population('test_fitness')
        fitness_list = self.new_population.get_fitness()
        test_fitness = [2.189059, 2.009438, 2.260596, 2.140625, 2.397067, 2.007199, 2.025928, 2.006774, 2.361584, 2.030199]
        for value in range(len(fitness_list)):
            self.assertEqual(round(fitness_list[value], 6), test_fitness[value])
        os.remove('test_fitness.json')
    
    def test_compare_individuals(self):
        result = self.new_population.compare_individuals(1 ,4)
        self.assertEqual(result, {'excess': 0, 'disjoint': 0, 'genome_size': 2, 'weight_mean': 2.2515})
        individuals_list = [
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 2.134), Connection(2, 2, 3, 1.145)]),
            Brain(2, 1, 1, 100, connection_list=[Connection(1, 1, 3, 0.122), Connection(2, 2, 3, 1.197, False), Connection(3, 2, 4, 1.345), Connection(4, 4, 3, 3.521)])
        ]
        self.new_population.load_population(individuals_list, 0, 0.0, 0.0, 0)
        result = self.new_population.compare_individuals(0, 1)
        self.assertEqual(result, {'excess': 2, 'disjoint': 1, 'genome_size': 3, 'weight_mean': 2.012})

    def test_speciation(self):
        for value in self.inputs_and_answers:
            self.new_population.set_inputs(self.inputs_and_answers[value][0])
            self.new_population.run_simulation()
            self.new_population.calculate_fitness(my_fitness, self.inputs_and_answers[value][1])
        self.new_population.save_population('test_fitness')
        self.new_population.speciation()
        self.assertEqual(len(self.new_population.get_species()), 1)
        self.new_population.load_population(self.individuals_list, 0, 0.0, 0.0, 0, threshold=0.5)
        self.new_population.speciation()
        self.assertGreater(len(self.new_population.get_species()), 1)
    
    def test_set_best_individual(self):
        for value in self.inputs_and_answers:
            self.new_population.set_inputs(self.inputs_and_answers[value][0])
            self.new_population.run_simulation()
            self.new_population.calculate_fitness(my_fitness, self.inputs_and_answers[value][1])
        self.new_population.set_best_individual()
        self.assertEqual(self.new_population.get_best_individual_info(), [4, 2.3970670000000003])

class PopulationTestCase2(unittest.TestCase):
    def setUp(self) -> None:
        self.brain_settings={
            "INPUTS": 2,
            "HIDDEN": 0,
            "OUTPUTS": 1,
            "CONNECTIONS": 100
        }
        self.new_population = Population(10, self.brain_settings, {}, False, False)
        self.individuals_list = [
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 2.134), Connection(2, 2, 3, 1.145)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.122), Connection(2, 2, 3, 1.197)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 1.314), Connection(2, 2, 3, 3.228)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 2.595), Connection(2, 2, 3, 0.733)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 3.157), Connection(2, 2, 3, 2.665)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.044), Connection(2, 2, 3, 2.227)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.198), Connection(2, 2, 3, 1.742)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.742), Connection(2, 2, 3, 0.178)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 1.967), Connection(2, 2, 3, 3.982)]),
            Brain(2, 0, 1, 100, connection_list=[Connection(1, 1, 3, 0.435), Connection(2, 2, 3, 0.994)])
        ]
        self.new_population.load_population(self.individuals_list, 0, 0.0, 0.0, 0)
        self.inputs_and_answers = {
            "IP1": [[0, 0], [0]],
            "IP2": [[1, 0], [1]],
            "IP3": [[0, 1], [1]],
            "IP4": [[1, 1], [0]]
        }
    
    def test_ajusted_fitness(self):
        pass

    def test_population_fitness(self):
        pass

    def test_calculate_offspring(self):
        pass

    def test_pick_one(self):
        pass

    def test_crossover(self):
        pass
        # for value in self.inputs_and_answers:
        #     self.new_population.set_inputs(self.inputs_and_answers[value][0])
        #     self.new_population.run_simulation()
        #     self.new_population.calculate_fitness(my_fitness, self.inputs_and_answers[value][1])
        # self.new_population.speciation()
        # total_offspring = 0
        # for specie in self.new_population.get_species_objects():
        #     total_offspring += int(specie.get_offspring())
        # self.assertEqual(total_offspring, self.new_population.get_info()['popsize'])
        # self.new_population.crossover()

    def test_mutate_change_weights(self):
        pass

    def test_mutate_add_connection(self):
        pass

    def test_mutate_add_node(self):
        pass

    def test_change_connection_state(self):
        pass

    def test_change_node_state(self):
        pass


if __name__ == '__main__':
    unittest.main()