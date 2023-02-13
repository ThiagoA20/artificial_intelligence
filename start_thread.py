import threading
import time
from draw_network import *
from main_neat import *
import global_vars


def my_fitness(output_list: list, answers: list) -> float:
    "output list and answers must be the same size, and the output of this function must be positive"
    error = round(abs(answers[0] - output_list[0]), 6)
    fitness = round(1 - error, 6)
    return fitness


inputs_and_answers = {
    "IP1": [[0, 0], [0]],
    "IP2": [[1, 0], [1]],
    "IP3": [[0, 1], [1]],
    "IP4": [[1, 1], [0]]
}

def main(population):
    for i in range(1000):
        for input_value in inputs_and_answers:
            population.set_inputs(inputs_and_answers[input_value][0])
            population.run_simulation()
            population.calculate_fitness(my_fitness, inputs_and_answers[input_value][1])
        population.speciation()
        population.crossover()
        population.mutate()

        population.draw_fittest_network()
        time.sleep(0.1)

my_population = Population(
    popsize=10, 
    brain_settings={
        "INPUTS": 2,
        "HIDDEN": 0,
        "OUTPUTS": 1,
        "CONNECTIONS": 100
    }, 
    mutate_probs={
        "connection_weight": 0.8,
        "add_connection": 0.05, 
        "add_node": 0.1,
        # "add_node": 0.01,
        "connection_state": 0.01,
        "node_state": 0.01
    },
    allow_bias=False, 
    allow_recurrency=False,
    threshold=6.0,
    threshold_change_ratio=0.3
)

brain_analyser_thread = threading.Thread(target=brain_analyser)
brain_analyser_thread.start()

brain_thread = threading.Thread(target=main, args=(my_population,), daemon=True)
brain_thread.start()

brain_analyser_thread.join()
# print(threading.active_count())