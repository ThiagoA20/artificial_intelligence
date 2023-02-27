"""
Problema:
Dados dois números, se a soma deles for maior que 1, o resultado é 1, caso contrário, o resultado é 0.

Solução:
Uma rede neural de topologia fixa (2 neurônios na camada de entrada, 3 na camada oculta e 1 na de saída) será utilizada para fazer
o feedfoward dos dados. 

As conexões estão feitas de modo que o primeiro neurônio da primeira camada vai estar conectado a um neurônio da camada oculta sozinho,
conectado com um neurônio da camada oculta em comum com o outro neurônio de entrada e o outro neurônio está da mesma forma, conectado ao neurônio do meio em comum com o primeiro neurônio de entrada e a outro neurônio da camada oculta sozinho. Essa configuração permite que seja analizado a influência das entradas isoladas e juntas no resultado final.

A otimização da rede será feita com um algoritmo genético, o fitness calculado de um indivíduo vai determinar a sua probabilidade de ser escolhido para o crossover. Depois que dois indivíduos são escolhidos, um novo indivíduo é criado pegando pesos de forma aleatória de cada um dos parentes, depois ele é adicionado à uma lista de novos indivíduos e esse processo é repetido até que o tamanho da nova lista de indivíduos seja igual ao tamanho da população. Feito isso, as mutações são aplicadas aos indivíduos e então a nova população é definida e todo o processo de testes começa de novo.

O melhor indivíduo encontrado é apenas copiado, não sofre crossover e nem mutações.
"""

import random
import math
import copy

# Gera um conjunto de dados com duas variáveis de entrada (x1 e x2) e uma variável de saída (y)
def generate_data(n):
    data = []
    for i in range(n):
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        if x1 + x2 > 1:
            y = 1
        else:
            y = 0
        data.append((x1, x2, y))
    return data

# Gera um conjunto de dados de treinamento e um conjunto de dados de teste
train_data = generate_data(400)
test_data = generate_data(100)

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Classe que define a rede neural
class NeuralNetwork:
    def __init__(self, weights: list = []):
        if weights == []:
            self.weights1 = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
            self.weights2 = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(1)]
        else:
            self.weights1 = [[weights[0], weights[1]], [weights[2], weights[3]]]
            self.weights2 = [[weights[4], weights[5], weights[6]]]
        self.fitness = 0
    
    def get_weights(self) -> list[float]:
        weights = []
        for layer in self.weights1:
            for value in layer:
                weights.append(value)
        for layer in self.weights2:
            for value in layer:
                weights.append(value)
        return weights
    
    def set_fitness(self, fitness: float) -> None:
        self.fitness += fitness
    
    def get_fitness(self) -> float:
        return self.fitness

    def forward(self, input_values: tuple[float]) -> list[float]:
        """
        Função de propagação que calcula as saídas dos neurônios, recebe duas entradas correspondentes aos dois neurônios
        4 conexões da camada de entrada com a camada oculta | formato: [[1, 1], [1, 1]]
        - a propagação da primeira camada para a segunda gera 3 saídas.
        3 conexões da camada oculta com a camada de saída   | formato: [[1, 1, 1]]        
        - a propagação da segunda para a terceira camada gera 1 saída.
        """
        h1 = [
            sigmoid(input_values[0] * self.weights1[0][0]),
            sigmoid(sum([input_values[0] * self.weights1[0][1], input_values[1] * self.weights1[1][0]])),
            sigmoid(input_values[1] * self.weights1[1][1])
        ]
        h2 = [sigmoid(sum([h1[j] * self.weights2[0][j] for j in range(3)]))]
        return h2[0]
    
    # Função que realiza a mutação do indivíduo
    def mutate(self) -> None:
        weights = self.get_weights()
        for i in range(len(weights)):
            mutate_prob = random.uniform(0.0, 1.0)
            if mutate_prob <= 0.8:
                mutation_type = random.uniform(0.0, 1.0)
                if mutation_type <= 0.9:
                    weights[i] = random.uniform(weights[i] * 0.8, weights[i] * 1.2)
                else:
                    weights[i] = random.uniform(-1, 1)
        self.weights1 = [[weights[0], weights[1]], [weights[2], weights[3]]]
        self.weights2 = [[weights[4], weights[5], weights[6]]]

# Função que calcula o fitness do indivíduo
def calculate_fitness(answer: int, output: float) -> float:
    error = abs(answer - output)
    fitness = round(1 - error, 6)
    return fitness

# Função que escolhe um indivíduo com base no fitness
def pool_selection(individuals_info: dict, sum_fitness: float) -> int:
    individual_id = -1
    r = random.uniform(0, sum_fitness)
    list_keys = list(individuals_info.keys())
    list_values = list(individuals_info.values())
    while r >= 0:
        individual_id += 1
        r -= list_values[individual_id]
    return list_keys[individual_id]

# Função que realiza o cruzamento entre dois indivíduos
def crossover(parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
    parent1_weights = parent1.get_weights()
    parent2_weights = parent2.get_weights()
    new_weights = []
    for i in range(len(parent1_weights)):
        selected_weight = random.randint(1, 2)
        if selected_weight == 1:
            new_weights.append(parent1_weights[i])
        else:
            new_weights.append(parent2_weights[i])
    return NeuralNetwork(new_weights)

# Algoritmo de treinamento utilizando o algoritmo genético
def train(train_data: list[tuple], population_size: int = 100, num_generations: int = 1000):
    # Cria a população inicial de indivíduos
    population = [NeuralNetwork() for _ in range(population_size)]
    best_individual = -1
    for generation in range(num_generations):
        for data in train_data:
            for network in population:
                output = network.forward((data[0], data[1]))
                fitness = calculate_fitness(data[2], output)
                network.set_fitness(fitness)
        individuals_info = {i: population[i].get_fitness() for i in range(len(population))}
        sum_fitness = sum(list(individuals_info.values()))
        best_individual = max(individuals_info, key=individuals_info.get)
        new_population = []
        for i in range(population_size):
            if i == best_individual:
                new_population.append(copy.deepcopy(population[best_individual]))
                new_population[i].fitness = 0
            else:
                parent1 = pool_selection(individuals_info, sum_fitness)
                parent2 = pool_selection(individuals_info, sum_fitness)
                new_population.append(crossover(population[parent1], population[parent2]))
                new_population[i].mutate()
        print(f"Generation: {generation} | best individual: {best_individual} | best_fitness: {population[best_individual].get_fitness()}")
        population = new_population
    return population[best_individual]

def test(network: NeuralNetwork, test_data: list[tuple]) -> str:
    total_score = 0
    result_score = 0
    for data in test_data:
        output = network.forward((data[0], data[1]))
        total_score += 1
        result_score += calculate_fitness(data[2], output)
    accuracy = f'Precisão obtida: {(100 * result_score)/total_score}%'
    return accuracy

# Testa a rede neural treinada com alguns exemplos
nn = train(train_data)
print(test(nn, test_data))

print("\nDados dois números, se a soma deles for maior que 1, o resultado é 1, caso contrário, o resultado é 0.")
print(f"[0.1, 0.1] => Esperado: 0.0 | Obtido: {nn.forward([0.1, 0.1])}")
print(f"[0.5, 0.0] => Esperado: 0.0 | Obtido: {nn.forward([0.5, 0.0])}")
print(f"[0.0, 0.5] => Esperado: 0.0 | Obtido: {nn.forward([0.0, 0.5])}")
print(f"[0.9, 0.9] => Esperado: 1.0 | Obtido: {nn.forward([0.9, 0.9])}")
print(f"[1.0, 1.0] => Esperado: 1.0 | Obtido: {nn.forward([1.0, 1.0])}")