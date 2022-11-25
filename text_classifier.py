import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from text_preproc import TextPreproc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from dataclasses import dataclass


@dataclass(frozen=False)
class Parameters:
    # fundamental:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type: str = 'dense'

    # Preprocessing parameters:
    context_size: int = 2
    embedding_dim: int = 64

    # conv net parameters:
    out_size: int = 32
    padding: int = 0
    dilation: int = 1
    kernel_size: int = 2
    stride: int = 1

    # dense net parameters:
    out_size_1 = 128
    out_size_2 = 32

    # Training parameters:
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 0.001


class DenseNN(nn.Module):
    """3 layer dense neural network with embedding layer"""

    def __init__(self, vocab_size, embedding_dim):
        super(DenseNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class ConvNN(nn.Module):
    """convolutional neural network with embedding layer"""

    def __init__(self):
        super(ConvNN, self).__init__()

    def forward(self, x):
        return x


class PandaSet(Dataset):
    """
    Dataset from pandas dataframe.
    Target column must be the last one
    """

    def __init__(self, data):
        super().__init__()

        x = data[data.columns[:-1]].values
        y = data[data.columns[-1]].values

        self.x = torch.tensor(x).to(torch.long)
        self.y = torch.tensor(y).to(torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NetworkMaster:
    """master class: training DenseNN model"""

    def __init__(self, data, vocab, params):

        self.params = params
        self.vocab = vocab
        self.data = data
        self.data[self.data.columns[0]] = self.text_to_vec(self.data[self.data.columns[0]])
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)

        self.device = self.params.device
        if params.model_type == 'dense':
            self.net_model = DenseNN(len(vocab), self.params.embedding_dim).to(self.device)
        elif params.model_type == 'conv':
            self.net_model = ConvNN().to(self.device)

        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.loss_fn = None

    def text_to_vec(self, text_corpus):
        return [[self.vocab[word] for word in text.split(' ')] for text in text_corpus]

    def ngram_preporc(self, text_corpus):
        """ngram preprocessing"""

        ngram_corpus = []
        for text in text_corpus:
            text = text.split(' ')
            ngram = [[[text[i - j - 1] for j in range(self.params.context_size)], text[i]]
                     for i in range(self.params.context_size, len(text))]
            ngram_corpus.append(ngram)

        return ngram_corpus

    def bow_preporc(self, text_corpus):

        """bag of words preprocessing"""

        bow_corpus = []
        for text in text_corpus:
            text = text.split(' ')
            bow = [[[text[i - j - 1] for j in range(self.params.context_size)] +
                    [text[i + j + 1] for j in range(self.params.context_size)], text[i]]
                   for i in range(self.params.context_size, len(text))]
            bow_corpus.append(bow)

        return bow_corpus

    def train_model(self):

        size = len(self.train_loader.dataset)
        self.net_model.train()

        for batch, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # loss between forward and real vals
            pred = self.net_model(x)
            loss = self.loss_fn(pred, y)

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_model(self):

        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        self.net_model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.net_model(x)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def fit(self, epochs=10, batch_size=1, learning_rate=1e-3):

        self.train_loader = DataLoader(PandaSet(self.train_data), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(PandaSet(self.test_data),
                                      batch_size=batch_size,
                                      shuffle=True)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net_model.parameters(), lr=learning_rate)

        for num in range(epochs):
            print(f"Epoch {num + 1}\n-------------------------------")
            self.train_model()
            self.test_model()

        return self.net_model


if __name__ == "__main__":
    # setting the parameters:
    params = Parameters

    # preprocessing:
    preproc_model = TextPreproc()
    net_model = NetworkMaster(preproc_model.get_preprocd_data(), preproc_model.get_vocab(), params)
    #print(net_model.ngram_preporc(['Hello my dear friend', 'how are you']))

    # network training:
    net_model.fit(epochs=20, batch_size=1)
