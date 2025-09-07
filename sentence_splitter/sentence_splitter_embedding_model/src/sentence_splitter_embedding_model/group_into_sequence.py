SIZE = 3 # Number of words to put on each input of the encoder model

def group_into_sequences(df, seq_len=SIZE):
    tokens = df['token'].tolist()
    labels = df['label'].tolist()
    
    # Group into sequences of seq_len
    token_seqs = [tokens[i:i+seq_len] for i in range(0, len(tokens), seq_len)]
    label_seqs = [labels[i:i+seq_len] for i in range(0, len(labels), seq_len)]
    
    return {'tokens': token_seqs, 'labels': label_seqs}