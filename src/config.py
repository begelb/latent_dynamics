import ast

class Config:

    def __init__(self, config_fname):
        with open(config_fname) as f:
            config = ast.literal_eval(f.read())
        self.system = config['system']
        self.high_dim = config['high_dim']
        self.ex_index = config['ex_index']
        self.num_layers = config['num_layers']
        self.scaler_dir = config['scaler_dir']
        self.model_dir = config['model_dir']
        self.log_dir = config['log_dir']
        self.output_dir = config['output_dir']
        self.subdiv_limit = config['subdiv_limit']
        self.num_layers = config['num_layers']
        self.hidden_shape = config['hidden_shape']
        self.non_linearity = config['non_linearity']
        self.high_dims = config['high_dims']
        self.low_dims = config['low_dims']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.patience = config['patience']
        self.data_dir = config['data_dir']

