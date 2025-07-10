import torch

class ClassificationHead(torch.nn.Module):

    def __init__(self, class_size, embed_size):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size

        self.hidden = torch.nn.Linear(embed_size, embed_size)

        self.relu = torch.nn.ReLU()
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        hidden_state = self.hidden(hidden_state)

        hidden_state = self.relu(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


class RegressionHead(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.hidden = torch.nn.Linear(embed_size,embed_size)
        self.relu = torch.nn.ReLU()
        self.mlp = torch.nn.Linear(embed_size, 1)


    def forward(self, hidden_state):
        hidden_state = self.hidden(hidden_state)
        hidden_state = self.relu(hidden_state)
        output = self.mlp(hidden_state)
        return output
