import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from src.config import Config
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='coral.txt')
    parser.add_argument('--verbose',help='Print training output',action='store_true',default=True)

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    config = Config(config_fname)

    high_dim = config.high_dim

    train_data = np.loadtxt(os.path.join(config.data_dir, 'train'), delimiter=',', skiprows=1)

    x_train = train_data[:, :high_dim]
    y_train = train_data[:, high_dim:]

    all_training_data = np.vstack((x_train, y_train))

    print(f"Fitting a single scaler on combined data with shape: {all_training_data.shape}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_training_data)

    scaler_dir = config.scaler_dir

    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, 'scaler.gz'))

if __name__ == "__main__":
    main()