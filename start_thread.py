import threading
import time
from draw_network import *
from main_neat import *
import global_vars


def my_fitness(output_list: list, answers: list) -> float:
    "output list and answers must be the same size, and the output of this function must be positive"
    fitness = 0
    for i in range(len(output_list)):
        if output_list[i] >= 0.5:
            result = 1
        else:
            result = 0
    ###################################################################### O problema do fitness ta aqui!
        fitness += 1 - (answers[i] - result)
        # print(f"Correct answer: {answers[i]} / Output: {result} fitness: {format(fitness, 'f')}")
    ###################################################################### O problema do fitness ta aqui!
    return 1


inputs_and_answers = {
    "IP1": [[0, 0], [0]],
    "IP2": [[1, 0], [1]],
    "IP3": [[0, 1], [1]],
    "IP4": [[1, 1], [0]]
}

def main(population):
    for i in range(2):
        for input_value in inputs_and_answers:
            # print(f"before inputs: {population.max_fitness}")
            population.set_inputs(inputs_and_answers[input_value][0])
            # print(f"before run: {population.max_fitness}")
            population.run_simulation()
            # print(f"before calculate fitness: {population.max_fitness}")
            population.calculate_fitness(my_fitness, inputs_and_answers[input_value][1])
            # print(f"after calculate fitness: {population.max_fitness}")
        # print(f"before speciation: {population.max_fitness}")
        population.speciation()
        # print(f"Speciation: {population.pop_fitness}")
        # print(f"before crossover: {population.max_fitness}")
        population.crossover()
        # print(f"before mutate: {population.max_fitness}")
        population.mutate()
        # print(f"after mutate: {population.max_fitness}")
        print("end population")

        population.draw_fittest_network()
        # print(population.get_best_individual_layers())
        time.sleep(3)

my_population = Population(
    popsize=50, 
    brain_settings={
        "INPUTS": 2,
        "HIDDEN": 0,
        "OUTPUTS": 1,
        "CONNECTIONS": 100
    }, 
    mutate_probs={
        "connection_weight": 0.8,
        "add_connection": 0.05, 
        "add_node": 0.0005,
        # "add_node": 0.01,
        "connection_state": 0.01,
        "node_state": 0.01
    },
    allow_bias=False, 
    allow_recurrency=False
)

brain_analyser_thread = threading.Thread(target=brain_analyser)
brain_analyser_thread.start()

brain_thread = threading.Thread(target=main, args=(my_population,), daemon=True)
brain_thread.start()

brain_analyser_thread.join()
# print(threading.active_count())