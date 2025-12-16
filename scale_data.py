import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

''' TO DO: Generalize to 4D --> updated file is on Amarel? '''

if __name__ == "__main__":
    th1=29
    th2=th1
    th3=th1
    print("Leslie Model Parameters:")
    print(f"th1: {th1}, th2: {th2}, th3: {th3}")
    
    system = 'Leslie'
    if system == 'Leslie':
        train_data = np.loadtxt(f'data/Leslie/{th1}_{th2}_{th3}/2train.csv', delimiter=',', skiprows=1)
        test_data = np.loadtxt(f'data/Leslie/{th1}_{th2}_{th3}/2test.csv', delimiter=',', skiprows=1)
    elif system == 'arctan':
        train_data = np.loadtxt(f'data/arctan/train.csv', delimiter=',', skiprows=1)
        test_data = np.loadtxt(f'data/arctan/test.csv', delimiter=',', skiprows=1)

    high_dims = 3
    x_train = train_data[:, :high_dims]
    y_train = train_data[:, high_dims:]
    x_test = test_data[:, :high_dims]
    y_test = test_data[:, high_dims:]

    ''' TO DO: don't separate and then vstack again '''

    all_training_data = np.vstack((x_train, y_train))

    print(f"Fitting a single scaler on combined data with shape: {all_training_data.shape}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_training_data)

    if system == 'Leslie':
        output_dir = f'output/Leslie/{th1}_{th2}_{th3}/scalers/'
    elif system == 'arctan':
        output_dir = f'output/arctan/scalers/'

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.gz'))
