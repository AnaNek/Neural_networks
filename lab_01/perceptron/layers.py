from perceptron.neurons import *


class NeuronLayer:
    def __init__(self):
        self.neurons = []

    def reinit_weights(self):
        for neuron in self.neurons:
            neuron.reinit_weights()

    def solve(self, inputs):
        raise NotImplementedError

    def correct(self, expected_results):
        pass


class SNeuronLayer(NeuronLayer):
    def add_neuron(self, f_initialize, f_transform):
        neuron = SNeuron(f_initialize, f_transform)
        self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron, value in zip(self.neurons, inputs):
            results.append(neuron.solve(value))
        return results


class ANeuronLayer(NeuronLayer):
    def add_neuron(self, inputs_count, f_initialize, f_activate):
        neuron = ANeuron(f_initialize, f_activate)
        neuron.init_weights(inputs_count)
        self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron in self.neurons:
            results.append(neuron.solve(inputs))
        return results


class RNeuronLayer(NeuronLayer):
    def add_neuron(self, inputs_count, f_initialize, f_activate, learning_speed, bias):
        neuron = RNeuron(f_initialize, f_activate, learning_speed, bias)
        neuron.init_weights(inputs_count)
        self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron in self.neurons:
            results.append(neuron.solve(inputs))
        return results

    def correct(self, expected_results):
        for neuron, expected_result in zip(self.neurons, expected_results):
            neuron.correct(expected_result)
