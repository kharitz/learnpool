import torch.utils.data
from learnpool.input.batch import Batch


class DataLoader(torch.utils.data.DataLoader):
    """
    Pytorch DataLoader to load brain graphs in batches
    Batch is Adapted from PyTorch Geometric torch_geometric.data.batch
    Source: https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/batch.py#L8
 the
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=4,
                 follow_batch=[]):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda batch: Batch.from_data_list(batch))
