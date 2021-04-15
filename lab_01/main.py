from perceptron.perceptron import Perceptron
from perceptron.layers import *
from training_dataset import dataset, test_dataset, NUMBER_COUNT
import random

def train_perceptron():
    network = Perceptron()
    input_count = len(dataset[0].inputs)
    print('----------------------------')
    print('Generating layers')
    for _ in range(input_count):
        network.s_layer.add_neuron(None, lambda value: value)
    print('S-layer generated')

    a_neurons_count = 2 ** input_count - 1
    for position in range(a_neurons_count):
        neuron = ANeuron(None, lambda value: int(value >= 0))
        # инициализация весов нейронов А слоя
        neuron.input_weights = [
            random.choice([-1, 0, 1]) for i in range(input_count)
        ]
        neuron.calculate_bias()
        network.a_layer.neurons.append(neuron)
    print('A-layer generated')

    for _ in range(NUMBER_COUNT):
        network.r_layer.add_neuron(a_neurons_count, lambda: 0, lambda value: 1 if value >=0 else -1, 0.01, 0)
    print('R-layer generated')

    network.train(dataset)
    network.optimize(dataset)
    return network


def test_network(network):
    total_classifications = len(test_dataset) * len(test_dataset[0].results)
    misc = 0
    for data in test_dataset:
        results = network.solve(data.inputs)
        for result, expected_result in zip(results, data.results):
            if result != expected_result:
                misc += 1

    print('----------------------------')
    print(
        'Test accuracy: {:.2f}%'.format(
            float(total_classifications - misc) / total_classifications * 100
        )
    )


def main():
    network = train_perceptron()
    test_network(network)


if __name__ == '__main__':
    main()