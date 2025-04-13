from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, encoded_texts, labels):
        self.data = encoded_texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), self.labels[idx]

def collate_batch(batch):
    texts, labels = zip(*batch)
    offsets = [0] + [len(seq) for seq in texts[:-1]]
    offsets = torch.tensor(offsets).cumsum(dim=0)
    text_tensor = torch.cat(texts)
    label_tensor = torch.tensor(labels)
    return text_tensor, offsets, label_tensor
