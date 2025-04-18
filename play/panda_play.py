import pandas as pd

def main():
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Marie'],
        'Age': [25, 30, 35, 40, 20],
        'City': ['New York', 'Los Angeles', 'San Francisco', 'Chicago', 'Washington']
    }

    df = pd.DataFrame(data)
    print(df)
    df = df.sort_values('Age')
    print(df)
    subcategory_to_id = {row["Name"]: index for index, (_, row) in enumerate(df.iterrows())}
    print(subcategory_to_id)

if __name__ == "__main__":
    main()