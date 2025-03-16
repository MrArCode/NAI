from point import Point


def get_user_points():
    while True:
        try:
            print("\nEnter a point to classify (comma-separated values matching the dataset dimensions).")
            print("For example, for Iris dataset: 5.1,3.5,1.4,0.2")
            user_input = input("Enter new point: ")

            if ";" in user_input:
                points = [Point(list(map(float, point_str.split(","))), "New-flower")
                          for point_str in user_input.split(";")]
            else:
                points = [Point(list(map(float, user_input.split(","))), "New-flower")]

            return points

        except ValueError:
            print("Invalid input! Ensure all numbers are correctly formatted.")


def get_k_value():
    while True:
        try:
            k = int(input("Enter the number of nearest neighbors (k): "))
            if k > 0:
                return k
            print("k must be greater than 0.")
        except ValueError:
            print("Invalid input! Please enter a valid integer.")