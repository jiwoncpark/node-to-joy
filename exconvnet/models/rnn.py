import torch
import torch.nn as nn

__all__ = ['RNN']

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=5, num_layers=1):
        super().__init__()

        # takes last hidden state and transforms it into R, should
        # be point estimate
        self.last_layer = nn.Linear(hidden_size, 1)

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, X):
        """Do a forward pass on a packed padded sequence mini-batch.

        Parameters
        ----------
        X : torch.nn.utils.rnn.PackedSequence
            Packed sequence input

        Returns
        -------
        Y : torch.Tensor
            output tensor containing n real numbers if batch size is n
        """

        # expecting X to be packed padded sequence
        _, hn = self.rnn(X)  # we ignore first output because it contains all hidden states

        # now that hn is the hidden layer at timestep n,
        # we can compute the posterior conditioned on our
        # sequence, which is encoded by hn
        Y = self.last_layer(hn)
        return Y

