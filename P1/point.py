import heapq
import math

class Point:
    def __init__(self, coordinates, label):
        self.coordinates = coordinates
        self.label = label

    def __str__(self):
        return f"Coordinates: {self.coordinates}, Label: {self.label}"

    def classify(self, train_data_points, k):
        distances = {index: calculate_distance_between_two_points(self, pointer)
                     for index, pointer in enumerate(train_data_points)}

        n_smallest = heapq.nsmallest(k, distances.items(), key=lambda x: x[1])
        label_counts = {}

        for element in n_smallest:
            label = train_data_points[element[0]].label
            label_counts[label] = label_counts.get(label, 0) + 1

        label = max(label_counts, key=label_counts.get)

        return label

def create_points(train_data_list):
    return [Point(list(map(float, line.split(";")[:-1])), line.split(";")[-1])
            for line in train_data_list if line.strip()]

def calculate_distance_between_two_points(first_point, second_point):
    return math.sqrt(sum((coord1 - coord2) ** 2
                         for coord1, coord2 in zip(first_point.coordinates, second_point.coordinates)))