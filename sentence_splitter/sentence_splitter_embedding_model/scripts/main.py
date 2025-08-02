import pandas as pd


def main():
    train = pd.DataFrame({'token': ['I', 'am', 'Fabio', '.', 'Going', 'to', 'see', 'Mike', '.'],
                          'label': [0, 0, 0, 1, 0, 0, 0, 0, 1]})
    validation = pd.DataFrame({'token': ['I', 'am', 'Fabio', '.'], 'label': [0, 0, 0, 1]})


if __name__ == "__main__":
    main()
