import random

from P2.binary_perceptron import BinaryPerceptron


class MultiClassPerceptron:
    def __init__(self, dimension, classes, alpha):
        self.dimension = dimension
        self.classes = classes
        self.alpha = alpha

        self.perceptrons = {}
        for cls in classes:
            weights = [random.uniform(-5, 5) for _ in range(dimension)]
            threshold = random.uniform(-5, 5)
            self.perceptrons[cls] = BinaryPerceptron(weights, threshold, alpha, cls)

    def compute(self, input_vector):
        confidences = {}
        for cls, perceptron in self.perceptrons.items():
            confidences[cls] = perceptron.calculate_net(input_vector)

        return max(confidences, key=confidences.get)

    def learn(self, input_vector, actual_class):
        deltas = {}
        for cls, perceptron in self.perceptrons.items():
            deltas[cls] = perceptron.learn(input_vector, actual_class)

        return deltas


def get_alpha_from_user():
    while True:
        try:
            alpha = float(input("Enter learning rate (alpha): "))
            return alpha
        except ValueError:
            print("Invalid input! Make sure you enter a number.")


def calculate_accuracy(multiclass_perceptron, test_vectors):
    correct = 0
    total = len(test_vectors)

    correct_by_class = {cls: 0 for cls in multiclass_perceptron.classes}
    total_by_class = {cls: 0 for cls in multiclass_perceptron.classes}

    for vector in test_vectors:
        predicted = multiclass_perceptron.compute(vector.coordinates)
        total_by_class[vector.label] += 1

        if predicted == vector.label:
            correct += 1
            correct_by_class[vector.label] += 1

    accuracy = correct / total if total > 0 else 0

    accuracy_by_class = {}
    for cls in multiclass_perceptron.classes:
        accuracy_by_class[cls] = correct_by_class[cls] / total_by_class[cls] if total_by_class[cls] > 0 else 0

    return accuracy, accuracy_by_class