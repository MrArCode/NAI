import sys

def read_lines_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read().splitlines()
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)