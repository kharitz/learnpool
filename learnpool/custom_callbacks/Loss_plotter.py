import matplotlib
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class LossPlotter(object):

    def __init__(self, mylog_path="./log", mylog_name="training.log", myloss_names=["loss"],
                 mymetric_names=["accuracy"], cmb_plot=1):
        super(LossPlotter, self).__init__()
        self.log_path = mylog_path
        self.log_name = mylog_name
        self.loss_names = list(myloss_names)
        self.metric_names = list(mymetric_names)
        self.cmb_plot = cmb_plot
        if cmb_plot:
            plt_path = join(self.log_path, "plot")
            if not os.path.exists(plt_path):
                os.makedirs(plt_path)
                os.makedirs(join(plt_path, "train"))
                os.makedirs(join(plt_path, "valid"))
                os.makedirs(join(plt_path, "test"))
        else:
            if not os.path.exists(join(self.log_path, "plot")):
                os.makedirs(join(self.log_path, "plot"))

    def plotter(self):

        dataframe = pd.read_csv(join(self.log_path, self.log_name), skipinitialspace=True)

        for i in range(len(self.loss_names)):
            plt.figure(i)
            plt.plot(dataframe[self.loss_names[i]], label="train_" + self.loss_names[i])
            plt.legend()
            plt.savefig(join(self.log_path, "plot", "train", self.loss_names[i] + ".png"))
            plt.close()

            plt.figure(i)
            plt.plot(dataframe["val_" + self.loss_names[i]], label="val_" + self.loss_names[i])
            plt.legend()
            plt.savefig(join(self.log_path, "plot", "valid", self.loss_names[i] + ".png"))
            plt.close()

            plt.figure(i)
            plt.plot(dataframe["test_" + self.loss_names[i]], label="test_" + self.loss_names[i])
            plt.legend()
            plt.savefig(join(self.log_path, "plot", "test", self.loss_names[i] + ".png"))
            plt.close()

        for i in range(len(self.metric_names)):
            plt.figure(i + len(self.loss_names))
            plt.plot(dataframe[self.metric_names[i]], label="train_" + self.metric_names[i])
            plt.legend()
            plt.savefig(join(self.log_path, "plot", "train", self.metric_names[i] + ".png"))
            plt.close()

            plt.figure(i + len(self.loss_names))
            plt.plot(dataframe["val_" + self.metric_names[i]], label="val_" + self.metric_names[i])
            plt.legend()
            plt.savefig(join(self.log_path, "plot", "valid", self.metric_names[i] + ".png"))
            plt.close()

            plt.figure(i + len(self.loss_names))
            plt.plot(dataframe["test_" + self.metric_names[i]], label="test_" + self.metric_names[i])
            plt.legend()
            plt.savefig(join(self.log_path, "plot", "test", self.metric_names[i] + ".png"))
            plt.close()
