"""Computation of mean and std from online streams of batches

"""


class RunningStats:
    def __init__(self, loader_dict):
        """Computation of mean and std from online streams of batches

        Parameters
        ----------
        loader_dict : dict
            dict of callable functions that can get the desired data from each
            batch

        """
        self.loader_dict = loader_dict
        stats = dict()
        for k, _ in self.loader_dict.items():
            stats[f'{k}_mean'] = 0.0
            stats[f'{k}_var'] = 0.0
        self.stats = stats

    def update(self, batch, i):
        """Update `stats` for a new batch

        Parameters
        ----------
        batch : array or dict
            new batch of data whose data can be accessed by the functions in
            `loader_dict`
        i : int
            index indicating that the batch is the i-th batch

        """
        for k, func in self.loader_dict.items():
            new = func(batch)
            new_mean = new.mean(dim=0, keepdim=True)
            new_var = new.var(dim=0, unbiased=False, keepdim=True)
            self.stats[f'{k}_var'] += (new_var - self.stats[f'{k}_var'])/(i+1)
            self.stats[f'{k}_var'] += (i/(i+1)**2.0)*(self.stats[f'{k}_mean'] - new_mean)**2.0
            self.stats[f'{k}_mean'] += (new_mean - self.stats[f'{k}_mean'])/(i+1)
