import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from text_preproc import TextPreproc
from dataclasses import dataclass
import pandas as pd
import os


@dataclass(frozen=False)
class Parameters:
    # fundamentals:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type: str = 'dense'

    # Preprocessing parameters:
    context_size: int = 2
    embedding_dim: int = 64

    # net parameters:
    out_size: int = 32
    padding: int = 0
    dilation: int = 1
    kernel_size: int = 2
    stride: int = 1


class DenseNet(nn.Module):
    """3 layer dense neural network with embedding layer"""
    def __init__(self, vocab_len):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=vocab_len, out_features=vocab_len // 10, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=vocab_len // 10, out_features=vocab_len // 100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=vocab_len // 100, out_features=2, bias=True)
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

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return max(log_probs)



class PandaSet(Dataset):
    """
    Dataset from pandas dataframe.
    Target column must be the last one
    """

    def __init__(self, data):
        super().__init__()

        x = data[data.columns[:-1]].values
        y = data[data.columns[-1]].values

        self.x = torch.tensor(x).to(torch.float32)
        self.y = torch.tensor(y).to(torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NetworkMaster:
    """master class: training models"""

    def __init__(self, data, vocab, params):

        self.params = params
        self.vocab = vocab

        self.device = self.params.device
        if params.model_type == 'dense':
            self.data = self.text_to_vec(data)
            self.model = DenseNet(len(vocab)).to(self.device)
            self.loss_fn = nn.CrossEntropyLoss()
        elif params.model_type == 'embed':
            self.data = self.text_to_embeds_vec(data)
            self.model = EmbeddingDenseNet(len(vocab), self.params.embedding_dim).to(self.device)
            self.loss_fn = nn.NLLLoss()
        else:
            raise ValueError("No such a model type as {}".format(params.model_type))

        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None

    def text_to_vec(self, data):
        """vectorization of the text using bag of words method"""

        out = []
        for i, text in enumerate(data[data.columns[0]]):
            vec = [0] * len(self.vocab)
            for word in text.split(' '):
                try:
                    vec[self.vocab[word]] += 1
                except:
                    pass
            out.append(vec)

        out_df = pd.DataFrame(data=out, columns=[idx for idx in range(len(self.vocab))])
        out_df[data.columns[-1]] = [num for num in data[data.columns[-1]]]
        return out_df

    def text_to_embeds_vec(self, data):
        """standardizing of the word id's vector"""

        df = pd.read_csv('dataset/train.tsv', sep='\t')
        max_words_num = 0
        for text in df[df.columns[0]]:
            max_words_num = max(max_words_num, len(text.split(' ')))

        out = []
        for i, text in enumerate(data[data.columns[0]]):
            vec = [0] * max_words_num
            for j, word in enumerate(text.split(' ')):
                vec[j] = self.vocab[word]
            out.append(vec)

        out_df = pd.DataFrame(data=out, columns=[idx for idx in range(max_words_num)])
        out_df[data.columns[-1]] = [num for num in data[data.columns[-1]]]
        return out_df

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
        self.model.train()

        for batch, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # loss between forward and real vals
            pred = self.model(x)
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
        self.model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def fit(self, epochs=10, batch_size=1, learning_rate=1e-3, save_model=True):

        self.train_loader = DataLoader(PandaSet(self.train_data), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(PandaSet(self.test_data),
                                      batch_size=batch_size,
                                      shuffle=True)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        for num in range(epochs):
            print(f"Epoch {num + 1}\n-------------------------------")
            self.train_model()
            self.test_model()

        if save_model:
            torch.save(self.model.state_dict(), "{}.pth".format(self.params.model_type))
            print("Saved PyTorch Model State to {}.pth".format(self.params.model_type))

        return self.model

    def predict(self, data_path, output_name='test'):

        test_df_orig = pd.read_csv(data_path, sep='\t')

        test_df_copy = test_df_orig.copy(deep=True)
        test_df_copy[test_df_copy.columns[0]] = preproc_model.preproc_data(test_df_orig[test_df_orig.columns[0]])
        test_df_copy = net_model.text_to_vec(test_df_copy)

        y_pred = []
        for i in range(len(test_df_copy)):
            y_pred.append(self.model(
                torch.Tensor(test_df_copy.iloc[i][test_df_copy.columns[:-1]]).to(torch.float32).to(
                    params.device)).argmax().item())

        test_df_orig['is_fake'] = y_pred
        if not os.path.exists('output/'):
            os.mkdir('output/')

        test_df_orig.to_csv('output/{}.tsv'.format(output_name), sep='\t')
        return test_df_orig


if __name__ == "__main__":
    # setting the parameters:
    params = Parameters

    # preprocessing:
    preproc_model = TextPreproc()
    net_model = NetworkMaster(preproc_model.get_preprocd_data(), preproc_model.get_vocab(), params)

    # network training:
    net_model.fit(epochs=20, batch_size=1, save_model=True)

    # testing:
    print(net_model.predict('dataset/test.tsv'))
