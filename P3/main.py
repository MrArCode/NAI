import os
import re
import random
import math
import glob
from collections import Counter


class Perceptron:
    def __init__(self, features_size):
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(features_size)]
        self.normalize_weights()

    def normalize_weights(self):

        norm = math.sqrt(sum(w * w for w in self.weights))

        if norm > 0:
            self.weights = [w / norm for w in self.weights]

    def activate(self, inputs):

        norm = math.sqrt(sum(x * x for x in inputs))
        if norm > 0:
            normalized_inputs = [x / norm for x in inputs]
        else:
            normalized_inputs = inputs

        dot_product = sum(w * x for w, x in zip(self.weights, normalized_inputs))
        return dot_product

    def binary_activate(self, inputs):

        return 1 if self.activate(inputs) > 0 else 0

    def train(self, inputs, expected, learning_rate=0.1):

        prediction = self.binary_activate(inputs)

        error = expected - prediction

        if error != 0:

            norm = math.sqrt(sum(x * x for x in inputs))
            if norm > 0:
                normalized_inputs = [x / norm for x in inputs]
            else:
                normalized_inputs = inputs

            self.weights = [w + learning_rate * error * x for w, x in zip(self.weights, normalized_inputs)]

            self.normalize_weights()

        return error


class NeuralNetwork:
    def __init__(self):
        self.perceptrons = {}
        self.features_size = 0
        self.languages = []
        self.word_frequencies = {}

    def extract_features(self, text):

        text = text.lower()
        text = re.sub(r'[^a-zA-ZąćęłńóśźżäöüßáéíóúñçÁÉÍÓÚÑÇ0-9\s]', '', text)

        words = text.split()
        word_count = Counter(words)

        features = [0] * self.features_size
        for word, count in word_count.items():
            if word in self.word_frequencies:
                index = self.word_frequencies[word]
                features[index] = count

        print(f"Input text: '{text}'")
        print(f"Recognized words: {[word for word in words if word in self.word_frequencies]}")

        return features

    def build_vocabulary(self, training_data):

        all_words = set()
        for language, texts in training_data.items():
            words_in_language = set()
            for text in texts:
                text = text.lower()
                text = re.sub(r'[^a-zA-ZąćęłńóśźżäöüßáéíóúñçÁÉÍÓÚÑÇ0-9\s]', '', text)
                words = text.split()
                all_words.update(words)
                words_in_language.update(words)
            print(f"Words in {language}: {words_in_language}")

        self.features_size = len(all_words)
        self.word_frequencies = {word: i for i, word in enumerate(all_words)}
        print(f"Full vocabulary: {self.word_frequencies.keys()}")

    def train(self, epochs=100, learning_rate=0.1):

        self.languages = [folder for folder in os.listdir() if os.path.isdir(folder) and folder != '__pycache__']

        if not self.languages:
            print("No language folders found!")
            return

        print(f"Found languages: {', '.join(self.languages)}")

        training_data = {}
        for language in self.languages:
            training_data[language] = []

            file_pattern = os.path.join(language, "*.txt")
            files = glob.glob(file_pattern)
            print(f"Found {len(files)} files in {language} folder: {files}")

            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        text = file.read()
                        if text.strip():
                            training_data[language].append(text)
                            print(f"  Loaded file: {file_path} ({len(text)} characters)")
                        else:
                            print(f"  Warning: Empty file: {file_path}")
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")

        total_texts = sum(len(texts) for texts in training_data.values())
        if total_texts == 0:
            print("Error: No training texts were loaded! Check your files and folders.")
            return

        self.build_vocabulary(training_data)

        print(f"Vocabulary size: {self.features_size}")
        if self.features_size == 0:
            print("Error: Vocabulary is empty! Check if your text files contain any words.")
            return

        for language in self.languages:
            self.perceptrons[language] = Perceptron(self.features_size)

        for epoch in range(epochs):
            total_error = 0

            for language in self.languages:

                for text in training_data[language]:

                    features = self.extract_features(text)

                    for lang, perceptron in self.perceptrons.items():
                        expected = 1 if lang == language else 0
                        error = perceptron.train(features, expected, learning_rate)
                        total_error += abs(error)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Total error: {total_error}")

            if total_error == 0 and epoch > 0:
                print(f"Training completed at epoch {epoch}")
                break

        print("\nFinal perceptron weights:")
        for language, perceptron in self.perceptrons.items():
            word_weights = [(word, perceptron.weights[idx]) for word, idx in self.word_frequencies.items()]
            top_words = sorted(word_weights, key=lambda x: x[1], reverse=True)[:10]
            print(f"{language} - Top words: {top_words}")

    def classify(self, text):

        features = self.extract_features(text)

        outputs = {}
        for language, perceptron in self.perceptrons.items():
            outputs[language] = perceptron.activate(features)

        max_language = max(outputs, key=outputs.get)

        binary_outputs = {lang: 1 if lang == max_language else 0 for lang in self.languages}

        return max_language, outputs, binary_outputs


def main():
    nn = NeuralNetwork()

    print("Training the neural network...")
    nn.train(epochs=200, learning_rate=0.2)

    if nn.features_size == 0:
        print("Training failed! Program cannot continue.")
        return

    while True:
        print("\nWprowadź tekst do klasyfikacji (lub wpisz 'exit' aby zakończyć):")
        text = input()

        if text.lower() == 'exit':
            break

        language, raw_outputs, binary_outputs = nn.classify(text)

        print(f"\nSklasyfikowano jako: {language}")
        print("\nWartości wyjściowe perceptronów:")
        for lang, output in raw_outputs.items():
            print(f"{lang}: {output:.4f}")

        print("\nWynik działania selektora maksimum:")
        for lang, output in binary_outputs.items():
            print(f"{lang}: {output}")


if __name__ == "__main__":
    main()