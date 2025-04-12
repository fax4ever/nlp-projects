from dataset import DataAccess

def main():
    data = DataAccess()
    feature = data.feature(332)
    print(feature)

if __name__ == "__main__":
    main()