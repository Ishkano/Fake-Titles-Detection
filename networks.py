from torch import nn
import torch.nn.functional as f


class DenseNet(nn.Module):
    """3 layer dense neural network"""

    def __init__(self, input_len):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=input_len, out_features=input_len // 4, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=input_len // 4, out_features=input_len // 16, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=input_len // 16, out_features=2, bias=True)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class EmbeddingDenseNet(nn.Module):
    """dense neural network with embedding"""

    def __init__(self, vocab_len, embedding_dim):
        super(EmbeddingDenseNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_len, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, x):
        embeds = self.embeddings(x)
        out = f.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = f.log_softmax(out, dim=1)
        return max(log_probs)


class ConvNet(nn.Module):
    """2 convolutional layers, 2 dense layers"""

    def __init__(self, input_len):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=20,
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=1,
                               kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=93, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=2)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool1d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool1d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x
