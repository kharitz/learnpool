import time
import argparse
import json
import os
import numpy as np
import torch
import torch.optim as optm
import torch.nn.functional as func
import sys

from os.path import join
from tqdm import tqdm
from shutil import copyfile
from learnpool.input.graphLoader import GeometricDataset
from learnpool.input.dataloader import DataLoader
from learnpool.custom_callbacks.Loss_plotter import LossPlotter
from learnpool.custom_callbacks.Logger import Logger
from learnpool.models.learnpool import GCNet
from learnpool.utils.utils import bin_accuracy
sys.path.insert(0, './learnpool/input/')


def _get_config():
    parser = argparse.ArgumentParser(description="Main handler for training",
                                     usage="python ./train.py -j config.json -g 0")
    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    parser.add_argument('-g', '--gpu', help='Cuda Visible Devices', required=True)
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        config = json.loads(f.read())

    initial_weights = config['generator']['initial_epoch']
    directory = os.path.join(config['directories']['out_dir'],
                             config['directories']['ConfigName'],
                             'config', str(initial_weights))
    if not os.path.exists(directory):
        os.makedirs(directory)

    copyfile(args.json, os.path.join(config['directories']['out_dir'],
                                     config['directories']['ConfigName'],
                                     'config', str(initial_weights), 'config.json'))

    # Set the GPU flag to run the code
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return config


def main(config):
    device = torch.device("cuda")

    generator_config = config['generator']  # Model experiments total epochs and beginning epoch
    initial_epoch = generator_config['initial_epoch']  # O by default and otherwise 'N' if loading
    num_epochs = generator_config['num_epochs']  # Total Number of Epochs
    plt_sep = generator_config['plot_separate']  # Plot the train, valid and test separately: 0 or 1
    lamb = generator_config['set_lamda']  # Plot the train, valid and test separately: 0 or 1
    loss_up = generator_config['loss_up']  # Plot the train, valid and test separately: 0 or 1
    bsz = generator_config['batch_size']  # Batch size for training/testing loaders

    # Get the model architecture
    model_params = config['model_params'] 
    fin = model_params['fin'] # Input node features
    fou1 = model_params['fou1'] # Output node features for first GC block
    clus = model_params['clus'] # Number of clusters learned for first GC block
    fou2 = model_params['fou2'] # Output node features for second GC block
    hlin = model_params['hlin'] # Output of the first liner layer
    outp = model_params['outp'] # Number of output classes
    psudim = model_params['psudim'] # Dimension of the pseudo-coordinates

    optm_config = config['optimizer']
    b1 = optm_config['B1']  # B1 for Adam Optimizer: Ex. 0.9
    b2 = optm_config['B2']  # B2 for Adam Optimizer: Ex. 0.999
    lr = optm_config['LR']  # Learning Rate: Ex. 0.001

    directory_config = config['directories']
    out_dir = directory_config['out_dir']  # Path to save the outputs of the experiments
    config_name = directory_config['ConfigName']  # Configuration Name to Uniquely Identify this Experiment
    log_path = join(out_dir, config_name, 'log')  # Path to save the training log files
    main_path = directory_config['datafile']  # Full Path of the dataset. Folder contains train, valid and test
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(join(log_path, 'weights')):  # Path to save the weights of training
        os.makedirs(join(log_path, 'weights'))

    # Initialize the model, optimizer and data loader
    model = GCNet(fin=fin, fou1=fou1, clus=clus, fou2=fou2, hlin=hlin, outp=outp, psudim=psudim)  # Create the model
    model = model.to(device)
    compute_loss = torch.nn.BCEWithLogitsLoss()  # Loss function: Binary cross-entropy with logits
    optimizer = optm.Adam(model.parameters(), lr=lr, betas=(b1, b2))
    train_set = GeometricDataset('train', main_path)
    train_loader = DataLoader(train_set,
                              batch_size=bsz,
                              num_workers=4,
                              shuffle=True)
    valid_set = GeometricDataset('valid', main_path)
    valid_loader = DataLoader(valid_set,
                              batch_size=bsz,
                              num_workers=4,
                              shuffle=False)
    test_set = GeometricDataset('test', main_path)
    test_loader = DataLoader(test_set,
                             batch_size=bsz,
                             num_workers=4,
                             shuffle=False)

    if initial_epoch > 0:
        print("===> Loading pre-trained weight {}".format(initial_epoch - 1))
        weight_path = 'weights/model-{:04d}.pt'.format(initial_epoch - 1)
        checkpoint = torch.load(join(log_path, weight_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optm.load_state_dict(checkpoint['optimizer_state_dict'])

    def checkpoint(epc):
        w_path = 'weights/model-{:04d}.pt'.format(epc)
        torch.save(
            {'epoch': epc, 'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}, join(log_path, w_path))

    # setup our callbacks used to plot curves
    my_metric = ['Accuracy']
    my_loss = ['Loss']
    logger = Logger(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)
    ls_plt = LossPlotter(mylog_path=log_path, mylog_name="training.log",
                         myloss_names=my_loss, mymetric_names=my_metric, cmb_plot=plt_sep)

    def train(loader):
        lss_all = acc_all = 0
        model.zero_grad()
        model.train()

        for data in tqdm(loader):
            data.to(device)
            optimizer.zero_grad()

            out, reg = model(data)
            loss = (loss_up * compute_loss(out, func.one_hot(torch.LongTensor([data.sx.item()]), num_classes=2).float().cuda())) + (lamb * reg)
            acc = bin_accuracy(torch.max(out, 1)[1], data.sx)
            loss.backward()
            optimizer.step()

            lss_all += loss.item()
            acc_all += acc

        metric = np.array([lss_all / len(loader), acc_all / len(loader)])

        return metric

    def test(loader):
        lss_all = acc_all = 0
        model.eval()

        with torch.no_grad():
            for data in tqdm(loader):
                data.to(device)

                out, reg = model(data)
                loss = (loss_up * compute_loss(out, func.one_hot(torch.LongTensor([data.sx.item()]), num_classes=2).float().cuda())) + (lamb * reg)
                acc = bin_accuracy(torch.max(out, 1)[1], data.sx)

                lss_all += loss.item()
                acc_all += acc

            metric = np.array([lss_all / len(loader), acc_all / len(loader)])

        return metric

    print("===> Starting Model Training at Epoch: {}".format(initial_epoch))

    for epoch in range(initial_epoch, num_epochs):
        start = time.time()

        print("\n\n")
        print("Epoch:{}".format(epoch))

        train_metric = train(train_loader)
        print(
            "===> Training   Epoch {}: Loss = {:.4f}, Accuracy = {:.4f}".format(epoch, train_metric[0], train_metric[1]))
        val_metric = test(valid_loader)
        print("===> Validation Epoch {}: Loss = {:.4f}, Accuracy = {:.4f}".format(epoch, val_metric[0], val_metric[1]))
        test_metric = test(test_loader)
        print("===> Testing    Epoch {}: Loss = {:.4f}, Accuracy = {:.4f}".format(epoch, test_metric[0], test_metric[1]))

        logger.to_csv(np.concatenate((train_metric, val_metric, test_metric)), epoch)
        ls_plt.plotter()
        checkpoint(epoch)
        end = time.time()
        print("===> Epoch:{} Completed in {:.4f} seconds".format(epoch, end - start))

    print("===> Done Training for Total {:.4f} Epochs".format(num_epochs))


if __name__ == "__main__":
    main(_get_config())
