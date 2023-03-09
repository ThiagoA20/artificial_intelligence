import threading
import time
from draw_network import *
from main_neat import *
import global_vars
import matplotlib.pyplot as plt
import pandas as pd


def my_fitness(output_list: list, answers: list) -> float:
    "output list and answers must be the same size, and the output of this function must be positive"
    error = round(abs(answers[0] - output_list[0]), 6)
    fitness = round(1 - error, 6)
    return fitness


inputs_and_answers = [
    [[0, 0], [0]],
    [[1, 0], [1]],
    [[0, 1], [1]],
    [[1, 1], [0]]
]

# def main(population):
#     for i in range(10000):
#         for input_value in inputs_and_answers:
#             population.set_inputs(inputs_and_answers[input_value][0])
#             population.run_simulation()
#             population.calculate_fitness(my_fitness, inputs_and_answers[input_value][1])
#             # population.update_results(inputs_and_answers[input_value][0])
#         population.speciation()
#         population.crossover()
#         population.mutate()

#         population.draw_fittest_network()
        # time.sleep(0.1)

# my_population = Population(
#     popsize=100, 
#     brain_settings={
#         "INPUTS": 2,
#         "HIDDEN": 0,
#         "OUTPUTS": 1,
#         "CONNECTIONS": 100
#     }, 
#     mutate_probs={
#         "connection_weight": 0.8,
#         "add_connection": 0.05,
#         "add_node": 0.03,
#         "connection_state": 0.1,
#         "node_state": 0.01
#     },
#     allow_bias=False, 
#     allow_recurrency=False,
#     threshold=6.0,
#     threshold_change_ratio=0.3
# )

# Algoritmo de treinamento utilizando o algoritmo genético
def train(train_data: list[tuple], population_size: int = 100, num_generations: int = 10000):
    # Cria a população inicial de indivíduos
    # population = [NeuralNetwork() for _ in range(population_size)]
    population = Population(
        popsize=population_size, 
        brain_settings={
            "INPUTS": 2,
            "HIDDEN": 0,
            "OUTPUTS": 1,
            "CONNECTIONS": 100
        }, 
        mutate_probs={
            "connection_weight": 0.8,
            "add_connection": 0.05,
            "add_node": 0.03,
            "connection_state": 0.1,
            "node_state": 0.01
        },
        allow_bias=False, 
        allow_recurrency=False,
        threshold=6.0,
        threshold_change_ratio=0.3
    )

    best_fitness = []

    for i in range(num_generations):
        for data in train_data:
            population.set_inputs(data[0])
            population.run_simulation()
            population.calculate_fitness(my_fitness, data[1])
            # population.update_results(inputs_and_answers[input_value][0])
            
        best_individual_info = population.get_best_individual_info()
        best_fitness.append(best_individual_info[1])
        print(f"Generation: {i} | best individual: {best_individual_info[0]} | best_fitness: {best_individual_info[1]}")
        population.speciation()
        population.crossover()
        population.mutate()
        population.draw_fittest_network()
    
    return [population.get_best_individual_object(), best_fitness]

    ####
    # best_individual = -1
    # for generation in range(num_generations):
    #     for data in train_data:
    #         for network in population:
    #             output = network.forward((data[0], data[1]))
    #             fitness = calculate_fitness(data[2], output)
    #             network.set_fitness(fitness)
    #     individuals_info = {i: population[i].get_fitness() for i in range(len(population))}
    #     sum_fitness = sum(list(individuals_info.values()))
    #     best_individual = max(individuals_info, key=individuals_info.get)
    #     new_population = []
    #     for i in range(population_size):
    #         if i == best_individual:
    #             new_population.append(copy.deepcopy(population[best_individual]))
    #             new_population[i].fitness = 0
    #         else:
    #             parent1 = pool_selection(individuals_info, sum_fitness)
    #             parent2 = pool_selection(individuals_info, sum_fitness)
    #             new_population.append(crossover(population[parent1], population[parent2]))
    #             new_population[i].mutate()
    #     print(f"Generation: {generation} | best individual: {best_individual} | best_fitness: {population[best_individual].get_fitness()}")
    #     population = new_population
    # return population[best_individual]

# def test(network: NeuralNetwork, test_data: list[tuple]) -> str:
#     total_score = 0
#     result_score = 0
#     for data in test_data:
#         output = network.forward((data[0], data[1]))
#         total_score += 1
#         result_score += calculate_fitness(data[2], output)
#     accuracy = f'Precisão obtida: {(100 * result_score)/total_score}%'
#     return accuracy

# Testa a rede neural treinada com alguns exemplos
# nn = train(tr)
# print(test(nn, test_data))

if __name__ == '__main__':
    brain_analyser_thread = threading.Thread(target=brain_analyser)
    brain_analyser_thread.start()

    # brain_thread = threading.Thread(target=main, args=(my_population,), daemon=True)
    brain_thread = threading.Thread(target=train, args=(inputs_and_answers,), daemon=True)
    brain_thread.start()

    brain_analyser_thread.join()
    # print(threading.active_count())