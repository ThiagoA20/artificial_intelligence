from threading import Lock

running = True
running_lock = Lock()

# counter = 0
# counter_lock = Lock()

network = [
    {'1': [1, 2], '2': [5], '3': [4], '4': [3]}, 
    {'1|4': [False, 3.251], '1|5': [True, 5.123], '5|4': [True, 2.152], '2|4': [True, 1.321], '4|3': [True, 1.742]},
    {'1': [1], '2': [1], '3': [0], '4': [0], '5': [0]}
]
network_lock = Lock()

network_info = {
    "individuals": 50,
    "species": 50,
    "generation": 100,
    "best_individual": 12,
    "best_fitness": 4.32,
    "threshold": 100.0,
    "connection_weight": 80,
    "add_connection": 0.5,
    "add_node": 0.01,
    "connection_state": 0.1,
    "node_state": 0.1,
    "allow_bias": False,
    "allow_recurrency": False,
    "input_neurons": 2,
    "hidden_neurons": 0,
    "output_neurons": 1
}
network_info_lock = Lock()