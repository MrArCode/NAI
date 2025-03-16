import matplotlib.pyplot as plt

def plot_data(train_points, test_points):
    train_x = [point.coordinates[2] for point in train_points]
    trina_y = [point.coordinates[3] for point in train_points]
    labels = [point.label for point in train_points]

    colors_dict = {
        "Iris-setosa": "yellow",
        "Iris-versicolor": "lightgreen",
        "Iris-virginica": "orange"
    }

    colors = [colors_dict.get(label, "blue") for label in labels]

    plt.figure(figsize=(10, 10))
    plt.scatter(train_x, trina_y, c=colors, marker="o", s=100, edgecolors='black')

    test_x = [point.coordinates[2] for point in test_points]
    test_y = [point.coordinates[3] for point in test_points]
    plt.scatter(test_x, test_y, c="red", marker="o", s=150, edgecolors='black')

    plt.title("Iris Dataset with New Point Classification")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
               for label, color in colors_dict.items()]
    handles.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Test Points'))
    plt.legend(handles=handles, title="Flower Type")

    plt.show()


def plot_accuracy_vs_k(train_data, test_data, k_values):
    accuracies = []
    for k in k_values:
        correct = 0
        for point in test_data:
            predicted = point.classify(train_data, k)
            if predicted == point.label:
                correct += 1
        accuracies.append(correct / len(test_data) * 100)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Accuracy vs k Value')
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()