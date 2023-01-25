import unittest
from main_neat import *
from math import ceil


class NeuronTestCase(unittest.TestCase):

    def test_initialize(self):
        # 0: HIDDEN, 1: SENSOR, 2: OUTPUT, 3: BIAS
        new_neuron = Neuron(neuron_id=1, neuron_type=1, neuron_layer=1)
        self.assertEqual(new_neuron.get_type(), 1)
        self.assertEqual(new_neuron.get_id(), 1)
        self.assertEqual(new_neuron.get_layer(), 1)
        self.assertEqual(new_neuron.get_neuron_info(), {"id": 1, "type": 1, "layer": 1, "Sum result": 0, "Activation result": 0})
    
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
    
    def test_calculate_activation_function(self):
        new_neuron = Neuron(neuron_id=1, neuron_type=2, neuron_layer=1)
        new_neuron.calculate_sum([-2.0, 5.4, 3.25, 1.2, -4.2])
        new_neuron.activate_neuron()
        self.assertEqual(new_neuron.get_output(), 0.9746672967731284)
        new_neuron = Neuron(neuron_id=1, neuron_type=1, neuron_layer=1)
        new_neuron.calculate_sum([7])
        new_neuron.activate_neuron()
        self.assertEqual(new_neuron.get_output(), 7)


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

    def test_add_connection(self):
        self.new_brain.add_connection()
        self.assertEqual(len(self.new_brain.get_connection_list()), 13)

    def test_layers_3_2_1_1(self):
        pass

    def test_add_node(self):
        print(self.new_brain.add_node())

    def test_mutate_weights(self):
        pass

class SpecieTestCase(unittest.TestCase):

    def test_1(self):
        pass

class PopulationTestCase(unittest.TestCase):

    def test_initialize_from_file(self):
        pass

    def test_crossover(self):
        pass

    def test_calculate_fitness(self):
        pass


if __name__ == '__main__':
    unittest.main()