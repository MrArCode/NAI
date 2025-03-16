import argparse
import sys

from P1.vizualization import plot_data, plot_accuracy_vs_k
from point import create_points
from user_input import get_user_points


sys.argv = ['main.py', '8', 'train-data', 'test-data']

def parse_arguments():
    parser = argparse.ArgumentParser(description='k-NN Classification')
    parser.add_argument('k', type=int, help='Number of nearest neighbors')
    parser.add_argument('train_set', help='Training set CSV file')
    parser.add_argument('test_set', help='Test set CSV file')
    return parser.parse_args()


def main():
    args = parse_arguments()
    k = args.k

    with open(args.train_set, "r", encoding="utf-8") as train_file:
        train_data_list = train_file.read().splitlines()

    train_data_points = create_points(train_data_list)

    with open(args.test_set, "r", encoding="utf-8") as test_file:
        test_data_list = test_file.read().splitlines()

    test_data_points = create_points(test_data_list)

    correct_predictions = 0
    for tested_point in test_data_points:
        calculated_label = tested_point.classify(train_data_points, k)
        print(
            f"Coordinates: {tested_point.coordinates}, Actual Label: {tested_point.label}, Predicted Label: {calculated_label}")
        if calculated_label == tested_point.label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data_points) * 100

    print(f"Accuracy of the model is: {accuracy}%")

    plot_data(train_data_points, test_data_points)

    k_values = range(1, 21)
    plot_accuracy_vs_k(train_data_points, test_data_points, k_values)

    while True:
        user_points = get_user_points()
        for point in user_points:
            print(
                f"Point with coordinates {point.coordinates} was classified as {point.classify(train_data_points, k)}")

        decision = ""
        while decision not in ['y', 'n']:
            decision = input("Would you like to check another point? (y/n): ").strip().lower()
            if decision == "y" or decision == "n":
                break
            else:
                print("Please enter 'y' or 'n'.")

        if decision == "n":
            break


if __name__ == "__main__":
    main()