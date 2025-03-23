class Vector:
    def __init__(self, coordinates, label):
        self.coordinates = coordinates
        self.label = label

    def __str__(self):
        return f"Coordinates: {self.coordinates}, Label: {self.label}"


def create_vectors(data_list):
    vectors = []
    for line in data_list:
        if line.strip():
            parts = line.split(";")
            coordinates = list(map(float, parts[:-1]))
            label = parts[-1]
            vectors.append(Vector(coordinates, label))
    return vectors