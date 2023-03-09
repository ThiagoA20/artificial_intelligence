import matplotlib.pyplot as plt
import pandas as pd
from train_nn import *

def get_fitness_in_function_of_generation(generations: int = 200, samples: int = 10):
    list_of_lists = []
    for i in range(samples):
        list_of_lists.append(train(inputs_and_answers, num_generations=generations)[1])
    generations = [i for i in range(generations)]
    for list in list_of_lists:
        plt.plot(generations, list)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.ylim(0.0, 4.0)
    plt.xlim(-5, generations + 5)
    plt.title('PSZ[100]NG[500]CW[0.8]AC[0.05]AN[0.03]CS[0.1]NS[0.01]')
    plt.savefig('5_psz100ng200cw80ac5an3cs10ns1.png')

get_fitness_in_function_of_generation()
plt.show()

"""
Plotar a média de tempo que uma geração permanece viva
Plotar a média de nodes da população em função da geração
"""