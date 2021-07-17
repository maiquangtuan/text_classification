import torch
from torch import nn 


class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class, n_layers, 
                bidirectional, dropout, pad_idx):
                super(LSTM_model, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional =bidirectional,
                dropout = dropout, batch_first = True)

                self.mlp = nn.Linear(hidden_dim * n_layers, num_class)


    def forward(self, text):
        
        embed_text = self.embedding(text)
        output, (hidden, cell) = self.lstm(embed_text)
        hidden = torch.cat([hidden[i,:,:] for i in range(hidden.size(0))], dim = 1)

        return self.mlp(hidden)




