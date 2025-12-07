import torch
import os
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
from src.models import *

class Training:
    def __init__(self, config, train_loader, test_loader, verbose):
        self.encoder = Encoder(config)
        self.dynamics = LatentDynamics(config)
        self.decoder = Decoder(config)
        self.verbose = bool(verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.encoder.to(self.device)
        self.dynamics.to(self.device)
        self.decoder.to(self.device)

        self.dynamics_train_loader = train_loader
        self.dynamics_test_loader = test_loader

        self.reset_losses()

        self.dynamics_criterion = nn.MSELoss(reduction='mean')

        self.lr = config.learning_rate
        self.model_dir = config.model_dir
        self.log_dir = config.log_dir

    def save_models(self, subfolder='', suffix=''):
        save_path = os.path.join(self.model_dir, subfolder)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.encoder, os.path.join(save_path, 'encoder' + suffix + '.pt'))
        torch.save(self.dynamics, os.path.join(save_path, 'dynamics' + suffix + '.pt'))
        torch.save(self.decoder, os.path.join(save_path, 'decoder' + suffix + '.pt'))
    
    def load_models(self):
        self.encoder = torch.load(os.path.join(self.model_dir, 'encoder.pt'))
        self.dynamics = torch.load(os.path.join(self.model_dir, 'dynamics.pt'))
        self.decoder = torch.load(os.path.join(self.model_dir, 'decoder.pt'))
    
    def save_logs(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_dir, 'train_losses' + '.pkl'), 'wb') as f:
            pickle.dump(self.train_losses, f)
        
        with open(os.path.join(self.log_dir, 'test_losses' + '.pkl'), 'wb') as f:
            pickle.dump(self.test_losses, f)
    
    def reset_losses(self):
        self.train_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_total': []}
        self.test_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_total': []}
    
    def forward(self, x_t, x_tau):
        x_t = x_t.to(self.device)
        x_tau = x_tau.to(self.device)

        # z_t = E(x_t)
        z_t = self.encoder(x_t)

        # x_t_pred = D(E(x_t))
        x_t_pred = self.decoder(z_t)

        # z_tau = E(x_tau)
        z_tau = self.encoder(x_tau)

        # x_tau_pred = D(E(x_tau))
        # this variable does not get passed forward?
        x_tau_pred = self.decoder(z_tau)

        # z_tau_pred = latent_dynamics(E(x_t))
        z_tau_pred = self.dynamics(z_t)

        # x_tau_pred_dyn = D(latent_dynamics(E(x_t)))
        x_tau_pred_dyn = self.decoder(z_tau_pred)

        return (x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn)

    def dynamics_losses(self, forward_pass, weight):
        x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn = forward_pass

        # x_t_pred = D(E(x_t)) so this is the reconstruction loss for x_t
        loss_ae1 = self.dynamics_criterion(x_t, x_t_pred)

        # x_tau_pred_dyn = D(latent_dynamics(E(x_t))) so this is also a dynamics loss. Should x_tau_pred_dyn be x_tau_pred?
        loss_ae2 = self.dynamics_criterion(x_tau, x_tau_pred_dyn)

        # z_tau_pred = latent_dynamics(E(x_t)) and z_tau = E(x_tau) so this is the dynamics loss 
        loss_dyn = self.dynamics_criterion(z_tau_pred, z_tau)
        loss_total = loss_ae1 * weight[0] + loss_ae2 * weight[1] + loss_dyn * weight[2]
        return loss_ae1, loss_ae2, loss_dyn, loss_total

    def train(self, epochs=1000, patience=50, weight=[1,1,1]):
        '''
        Function that trains all the models with all the losses and weight.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        print("Training with weights: ", weight)
        weight_bool = [bool(i) for i in weight]
        print('weight bool: ', weight_bool)

        list_parameters = (weight_bool[0] or weight_bool[1] or weight_bool[2]) * (list(self.encoder.parameters()) + list(self.decoder.parameters()))
        list_parameters += (weight_bool[1] or weight_bool[2]) * list(self.dynamics.parameters())
        optimizer = torch.optim.Adam(list_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=patience)
        for epoch in tqdm(range(epochs)):
            loss_ae1_train = 0
            loss_ae2_train = 0
            loss_dyn_train = 0

            epoch_train_loss = 0
            epoch_test_loss  = 0


            if weight_bool[0] or weight_bool[1] or weight_bool[2]: 
                # put encoder and decoder in train mode
                self.encoder.train() 
                self.decoder.train() 
            if weight_bool[1] or weight_bool[2]: 
                # put dynamics in train mode
                self.dynamics.train()

            num_batches = len(self.dynamics_train_loader)
            for (x_t, x_tau) in self.dynamics_train_loader:
                optimizer.zero_grad()

                # Forward pass (apply all models)
                forward_pass = self.forward(x_t, x_tau)
                # Compute losses
                loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)

                # Backward pass
                loss_total.backward()
                optimizer.step()

                loss_ae1_train += loss_ae1.item()
                loss_ae2_train += loss_ae2.item()
                loss_dyn_train += loss_dyn.item()

                epoch_train_loss += loss_total.item()

            epoch_train_loss /= num_batches

            self.train_losses['loss_ae1'].append(loss_ae1_train / num_batches)
            self.train_losses['loss_ae2'].append(loss_ae2_train / num_batches)
            self.train_losses['loss_dyn'].append(loss_dyn_train / num_batches)
            self.train_losses['loss_total'].append(epoch_train_loss)

            with torch.no_grad():
                loss_ae1_test = 0
                loss_ae2_test = 0
                loss_dyn_test = 0

                if weight_bool[0] or weight_bool[1] or weight_bool[2]:  
                    self.encoder.eval() 
                    self.decoder.eval() 
                if weight_bool[1] or weight_bool[2]: 
                    self.dynamics.eval()

                num_batches = len(self.dynamics_test_loader)
                for (x_t, x_tau) in self.dynamics_test_loader:
                    # Forward pass
                    forward_pass = self.forward(x_t, x_tau)
                    # Compute losses
                    loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)

                    loss_ae1_test += loss_ae1.item() 
                    loss_ae2_test += loss_ae2.item() 
                    loss_dyn_test += loss_dyn.item() 
                    epoch_test_loss += loss_total.item()

                epoch_test_loss /= num_batches

                self.test_losses['loss_ae1'].append(loss_ae1_test / num_batches)
                self.test_losses['loss_ae2'].append(loss_ae2_test / num_batches)
                self.test_losses['loss_dyn'].append(loss_dyn_test / num_batches)
                self.test_losses['loss_total'].append(epoch_test_loss)

            scheduler.step(epoch_test_loss)
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    if self.verbose:
                        print("Early stopping")
                    break
            l1 = (1/weight[0]) * (loss_ae1_train / num_batches) if weight[0] != 0 else 0
            l2 = (1/weight[1]) * (loss_ae2_train / num_batches) if weight[1] != 0 else 0
            l3 = (1/weight[2]) * (loss_dyn_train / num_batches) if weight[2] != 0 else 0
                
            if self.verbose:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss, epoch_test_loss))
                print('    Train Losses - AE1: {:.4f}, AE2: {:.4f}, Dyn: {:.4f}'.format(l1, l2, l3))
        return l1, l2, l3
    
    def fine_tune(self, epochs=1000, patience=50, weight=[0,1,1,0,0]):
        self.load_models()

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.train(epochs, patience, weight)
        



