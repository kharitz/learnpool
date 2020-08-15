import matplotlib
import pandas as pd
from os.path import join
matplotlib.use('Agg')

'''
A python class to log the loss and accuracy to a csv file
'''

class Logger(object):

    def __init__(self, mylog_path="./log", mylog_name="training.log", myloss_names=["loss"],
                 mymetric_names=["Accuracy"]):
        super(Logger, self).__init__()

        self.log_path = mylog_path
        self.log_name = mylog_name
        self.loss_names = list(myloss_names)
        self.metric_names = list(mymetric_names)

    def to_csv(self, metric_array, epoch):
        if epoch == 0:
            train_c = self.loss_names + self.metric_names
            val_c = ['val_' + t for t in train_c]
            test_c = ['test_' + t for t in train_c]
            df = pd.DataFrame(columns=train_c + val_c + test_c)
            df.loc[0] = metric_array
        else:
            df = pd.read_csv(join(self.log_path, self.log_name), index_col=0)
            df.loc[len(df)] = metric_array
        df.to_csv(join(self.log_path, self.log_name))
