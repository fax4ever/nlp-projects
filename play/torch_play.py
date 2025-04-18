import torch
from seed import set_seed

def main():
    set_seed(42)
    val = torch.randint(0, 20, (4,))
    # ciao tensor([ 2,  7, 16, 14])
    print("ciao", val)

if __name__ == "__main__":
    main()