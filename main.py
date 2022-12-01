import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sentence_transformers import SentenceTransformer

from parameters import Parameters
from text_preproc import TextPreproc, BagOfWordsVectorizer
from networks import DenseNet, ConvNet
from datasets import PandaSet, MyDataset
import os


class NetworkMaster:
    """master class: training models"""

    def __init__(self, params):

        self.params = params
        self.preproc_model = None
        self.embed_model = None
        self.network_model = None
        self.loss_fn = None
        self.optimizer = None

    def fit(self, data_path, save_model=True):
        """fit neural network with obtained data"""

        self.preproc_model = TextPreproc(data_path=data_path)
        text, target = self.preproc_model.get_initial_data()

        if self.params.preproc_model_type == "default":
            self.embed_model = BagOfWordsVectorizer(vocab=self.preproc_model.get_vocab())
        else:
            self.embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=self.params.device)

        embeds = self.embed_model.encode(text)
        x_train, x_test, y_train, y_test = train_test_split(embeds, target, test_size=0.2)
        train_loader = DataLoader(MyDataset(x_train, y_train), batch_size=self.params.batch_size, shuffle=True)
        test_loader = DataLoader(MyDataset(x_test, y_test), batch_size=self.params.batch_size, shuffle=True)

        if self.params.preproc_model_type == "default":
            self.network_model = DenseNet(len(x_train[0])).to(self.params.device)
        else:
            self.network_model = ConvNet(len(x_train[0])).to(self.params.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network_model.parameters(),
                                          lr=self.params.learning_rate)

        for num in range(self.params.epochs):
            print(f"Epoch {num + 1}\n-------------------------------")
            self.train_model(train_loader)
            self.test_model(test_loader)

        if save_model:

            if not os.path.exists('output/'):
                os.mkdir('output/')

            torch.save(self.network_model.state_dict(), "output/{}.pth".format(self.params.network_model_type))
            print("Saved PyTorch Model State to output/{}.pth".format(self.params.network_model_type))

        return self.network_model

    def predict(self, data_path, show_statts=False):
        """make predictions on obtained data"""

        text_corpus, true_target = self.preproc_model.preproc(data_path=data_path)

        pred_target = []
        for embed in self.embed_model.encode(text_corpus):
            pred_target.append(self.network_model(torch.Tensor(embed).
                                                  to(torch.float32).
                                                  to(self.params.device)).argmax().item())

        if show_statts:
            print('accuracy: {}'.format(accuracy_score(true_target, pred_target)))
            print('recall: {}'.format(recall_score(true_target, pred_target)))
            print('precision: {}'.format(precision_score(true_target, pred_target)))
            print('f1: {}'.format(f1_score(true_target, pred_target)))

        return pred_target

    def train_model(self, loader):
        """network training"""

        size = len(loader.dataset)
        self.network_model.train()

        for batch, (x, y) in enumerate(loader):

            self.optimizer.zero_grad()
            x, y = x.to(self.params.device), y.to(self.params.device)

            # loss between forward and real vals
            pred = self.network_model(x)
            loss = self.loss_fn(pred, y)

            # backpropagation
            loss.backward()
            self.optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_model(self, loader):
        """network evaluating"""

        size = len(loader.dataset)
        num_batches = len(loader)
        self.network_model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.params.device), y.to(self.params.device)
                pred = self.network_model(x)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    # setting parameters:
    parameters = Parameters

    # training:
    model = NetworkMaster(params=parameters)
    model.fit(data_path='kinopoisk_data/train.csv', save_model=True)

    # testing:
    print(model.predict(data_path='kinopoisk_data/test.csv'))
