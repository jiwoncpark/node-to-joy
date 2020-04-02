import torch
import torch.nn as nn

__all__ = ['RNN']

class RNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=5):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, num_classes)

    def _forward_once(self, x, h, compute_out=False):
        """Compute a single step of the RNN forward pass.

        Parameters
        ----------
        x : torch.FloatTensor
            input example
        h : torch.FloatTensor
            the hidden state from the previous timestep
        compute_out : bool
            whether to compute the output

        Returns
        -------
        output : torch.Tensor
            the output point estimate of shape (1,1)
        hidden : torch.Tensor
            the next hidden state
        """

        combined = torch.cat((x, h))
        hidden = self.i2h(combined)

        if compute_out:
            output = self.i2o(combined)
        else:
            output = None

        return output, hidden

    def forward(self, X):
        # TODO figure out some way to do this
        # in a more efficient/parallel way

        out = torch.empty(X.shape[0])
        for i, x in enumerate(X):
            h = self.init_hidden()
            
            if x.shape[0] != 1:
                for x_i in x[:-1]:
                    _, h = self._forward_once(x_i, h)

            out[i] = self._forward_once(x[-1], h, compute_out=True)[0]

        return out

    def init_hidden(self):
        """Initialize the hidden state.

        Returns
        -------
        hidden : torch.Tensor
            the first hidden state
        """
        
        hidden = torch.zeros(self.hidden_size)
        return hidden
