class BinaryPerceptron:
    def __init__(self, weights, threshold, alpha, positive_class):
        self.weights = weights
        self.threshold = threshold
        self.alpha = alpha
        self.positive_class = positive_class

    def calculate_net(self, input_vector):
        return sum(i * w for i, w in zip(input_vector, self.weights)) - self.threshold

    def compute(self, input_vector):
        net = self.calculate_net(input_vector)
        return 1 if net >= 0 else 0

    def learn(self, input_vector, actual_class):
        expected = 1 if actual_class == self.positive_class else 0

        actual = self.compute(input_vector)

        delta = self.alpha * (expected - actual)

        for i in range(len(self.weights)):
            self.weights[i] += delta * input_vector[i]

        self.threshold -= delta

        return delta