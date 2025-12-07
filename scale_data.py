import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib # Import joblib
import os

if __name__ == "__main__":
    # num_pt_list = []
    # for j in range(11, 16):
    #     num_pts = (2**j) * 10
    #     num_pt_list.append(num_pts)

    # for num_pts in num_pt_list:#[20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]:

   # system = 'Leslie'
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
    # Separate inputs (x) and outputs (y)
    x_train = train_data[:, :high_dims]
    #print('x_train shape:', x_train.shape)
    y_train = train_data[:, high_dims:]
    x_test = test_data[:, :high_dims]
    y_test = test_data[:, high_dims:]

    all_training_data = np.vstack((x_train, y_train))

    print(f"Fitting a single scaler on combined data with shape: {all_training_data.shape}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_training_data)

    # --- 6. Save the Single Scaler ---
    # Give it a more general name

    # # --- START: Normalization ---
    # # Initialize scalers
    # x_scaler = StandardScaler()
    # y_scaler = StandardScaler()

    # # Fit scalers ONLY on the training data
    # x_train_scaled = x_scaler.fit_transform(x_train)
    # y_train_scaled = y_scaler.fit_transform(y_train)

    # --- END: Normalization ---
    if system == 'Leslie':
        output_dir = f'output/Leslie/{th1}_{th2}_{th3}/scalers/'
    elif system == 'arctan':
        output_dir = f'output/arctan/scalers/'

    os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists
    # joblib.dump(x_scaler, os.path.join(output_dir, 'x_scaler.gz'))
    # joblib.dump(y_scaler, os.path.join(output_dir, 'y_scaler.gz'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.gz'))
