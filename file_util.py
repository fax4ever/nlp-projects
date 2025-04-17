import os, pickle

def dump(file_name, result):
    # remove dump files if present
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'wb') as file:
        print("dumping", file_name)
        # noinspection PyTypeChecker
        pickle.dump(result, file)

def load(file_name):
    with open(file_name, 'rb') as file:
        print("loading", file_name)
        # noinspection PyTypeChecker
        return pickle.load(file)