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

    def test_initialize(self):
        new_brain = Brain(input_neurons=self.INPUT_NEURONS, hidden_neurons=self.HIDDEN_NEURONS, output_neurons=self.OUTPUT_NEURONS, connections_percentage=self.CONNECTIONS_PERCENTAGE)
        self.assertEqual(len(new_brain.get_neuron_list()), self.INPUT_NEURONS + self.HIDDEN_NEURONS + self.OUTPUT_NEURONS + 1)
        real_connections = lambda x: ceil(x * self.CONNECTIONS_PERCENTAGE / 100)
        if self.HIDDEN_NEURONS > 0:
            connection_amount = (self.INPUT_NEURONS * self.HIDDEN_NEURONS) + (self.HIDDEN_NEURONS * self.OUTPUT_NEURONS)
        else:
            connection_amount = (self.INPUT_NEURONS * self.OUTPUT_NEURONS)
        self.assertEqual(len(new_brain.get_connection_list()), real_connections(connection_amount))
    
    def test_initialize_from_file(self):
        pass

    def test_add_node(self):
        pass

    def test_add_connection(self):
        pass

    def test_set_neuron_layers(self):
        pass

    def test_mutate(self):
        pass

    def test_load_inputs(self):
        pass

    def test_run_network(self):
        pass


if __name__ == '__main__':
    unittest.main()