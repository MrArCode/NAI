import random
import sys

from P2.multi_class_perceptron import MultiClassPerceptron, get_alpha_from_user, \
    calculate_accuracy
from P2.vector import create_vectors
from utils import read_lines_from_file


def main():
    alpha = get_alpha_from_user()

    print("\nStarting program...")

    train_data = read_lines_from_file("train-data")
    test_data = read_lines_from_file("test-data")

    train_vectors = create_vectors(train_data)
    test_vectors = create_vectors(test_data)

    random.shuffle(train_vectors)

    all_classes = list(set(vector.label for vector in train_vectors))
    print(f"Found {len(all_classes)} classes: {', '.join(all_classes)}")

    if not train_vectors:
        print("Error: No training data found.")
        sys.exit(1)

    dimension = len(train_vectors[0].coordinates)
    multiclass_perceptron = MultiClassPerceptron(dimension, all_classes, alpha)

    continue_learning = True
    iteration = 0

    while continue_learning:
        iteration += 1
        print(f"\nStarting learning iteration {iteration}...")

        for vector in train_vectors:
            multiclass_perceptron.learn(vector.coordinates, vector.label)

        accuracy, accuracy_by_class = calculate_accuracy(multiclass_perceptron, test_vectors)

        print(f"\nOverall accuracy: {accuracy:.4f}")

        for i, cls in enumerate(all_classes):
            print(f"Accuracy for class {i + 1} ({cls}): {accuracy_by_class[cls]:.4f}")

        print("\nCurrent perceptron weights:")
        for cls, perceptron in multiclass_perceptron.perceptrons.items():
            print(f"\nClass: {cls}")
            for i, weight in enumerate(perceptron.weights):
                print(f"  w{i + 1} = {weight:.4f}")
            print(f"  Threshold = {perceptron.threshold:.4f}")

        choice = input("\nDo you want to perform another learning iteration? (y/n): ").lower()
        continue_learning = choice == 'y'

    print("\nYou can now enter vectors for classification.")
    print("Format: x1;x2;x3;x4 (e.g., 5.1;3.5;1.4;0.2)")
    print("To exit, type 'exit'")

    while True:
        user_input = input("\nEnter vector for classification: ")

        if user_input.lower() in ['exit', 'quit', 'q']:
            break

        try:
            coordinates = list(map(float, user_input.split(";")))

            if len(coordinates) != dimension:
                print(f"Error: Vector should have {dimension} coordinates.")
                continue

            predicted_class = multiclass_perceptron.compute(coordinates)

            print(f"Classification: {predicted_class}")

            print("Confidence values:")
            confidences = {cls: perceptron.calculate_net(coordinates)
                           for cls, perceptron in multiclass_perceptron.perceptrons.items()}

            for cls, confidence in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {confidence:.4f}")

        except ValueError:
            print("Error: Invalid data format. Please enter numerical values separated by semicolons.")

    print("Program finished.")


if __name__ == "__main__":
    main()