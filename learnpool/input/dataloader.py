import torch.utils.data
from learnpool.input.batch import Batch


class DataLoader(torch.utils.data.DataLoader):
    """
    Pytorch DataLoader to load brain graphs in batches
    learnpool.input.batch: Refers to torch_geometric.data.batch for simpler implementation
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
