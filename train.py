from src.models import *
from src.training import Training
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import torch
import random
from src.config import Config
import os
import joblib
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='coral_hybrid4.txt')
    parser.add_argument('--verbose',help='Print training output',action='store_true',default=True)
    parser.add_argument('--seed',help='Random seed for reproducibility',type=int,default=None)
    parser.add_argument('--output_subdir',help='Subdirectory under output/model/log dirs (e.g. seed_42)',type=str,default=None)
    parser.add_argument('--train_file',help='Training CSV base name (without .csv)',type=str,default='train')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    config_fname = args.config_dir + args.config

    config = Config(config_fname)

    if args.output_subdir is not None:
        subdir_root       = os.path.join(config.output_dir, args.output_subdir)
        config.output_dir = subdir_root
        config.model_dir  = os.path.join(subdir_root, 'models')
        config.log_dir    = os.path.join(subdir_root, 'logs')
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.log_dir,   exist_ok=True)
    # ex_index = config.ex_index

    base_data_dir = config.data_dir
    train_data_path = os.path.join(base_data_dir, args.train_file + '.csv')
    test_data_path = os.path.join(base_data_dir, 'test.csv')
    train_data = np.loadtxt(train_data_path, delimiter=',', skiprows=1)
    test_data = np.loadtxt(test_data_path, delimiter=',', skiprows=1)

    scaler_dir = os.path.join(config.scaler_dir, args.train_file)
    scaler_path = os.path.join(scaler_dir, 'scaler.gz')
    scaler = joblib.load(scaler_path)

    high_dims = config.high_dims
    x_train = train_data[:, :high_dims]
    y_train = train_data[:, high_dims:]
    x_test = test_data[:, :high_dims]
    y_test = test_data[:, high_dims:]

    x_train_scaled = scaler.transform(x_train)
    y_train_scaled = scaler.transform(y_train)
    x_test_scaled = scaler.transform(x_test)
    y_test_scaled = scaler.transform(y_test)

    train_dataset = TensorDataset(torch.tensor(x_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(x_test_scaled, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))

    batch_size = config.batch_size

    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer = Training(config, train_loader, test_loader, args.verbose)

    print('Number of epochs: ', config.epochs)
    
    ''' TO DO: Log weights / add to config file'''
    # for coral and Leslie3D, used weight=[10, 10, 1]
    l1, l2, l3 = trainer.train(config.epochs, config.patience, weight=[10, 10, 1])
    trainer.save_logs()
    trainer.reset_losses()
    trainer.save_models()

    filename = os.path.join(config.output_dir, 'final_losses.txt')
    with open(filename, "w") as f:
        f.write("--- Final losses ---\n")
        f.write(f"L1: {l1}\n")
        f.write(f"L2: {l2}\n")
        f.write(f"L3: {l3}\n")
    

if __name__ == "__main__":
    main()