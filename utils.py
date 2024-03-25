#!/usr/bin/env python3

# Copyright (C) 2024 Ashish Kumar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program in the file: gpl-3.0.text. 
# If not, see <http://www.gnu.org/licenses/>.

# system imports
import os
import datetime
import sys

# library imports
import torch
import numpy as np
from tensordict.tensordict import TensorDict
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

# local imports
from config.loader import load_config

class Utils:

    @staticmethod
    def load_config(config_path, obj):
        trainer_config = load_config(config_path)

        # PPO Trainer Configs
        for config in trainer_config:
            setattr(obj, "_" + config, trainer_config[config])   

    @staticmethod
    def create_tensor(state, device = torch.device("cuda:0")):
        state = np.expand_dims(state, axis=0)
        return torch.from_numpy(state).type(torch.float32).to(device)   
    
    @staticmethod
    def create_sars(states, next_states, actions, rewards, terminations, device = torch.device("cuda:0")):
        sars_dict = {}

        sars_dict["state"] = states.to(device)
        sars_dict["action"] = actions.to(device)
        sars_dict["reward"] = rewards.to(device)
        sars_dict["termination"] = terminations.to(device)
        sars_dict[("next", "state")] = next_states.to(device)
        
        sars_dict = TensorDict(sars_dict, states.shape[0], device=device)
        return sars_dict
    
    @staticmethod
    def convert_params_to_multivariate_normal(mu, lower_triangular_vector, cov_matrix_dim):
        
        # https://pytorch.org/docs/stable/distributions.html#multivariatenormal
        tril = torch.zeros(mu.shape[0], cov_matrix_dim, cov_matrix_dim, device=mu.device)
        tril_indices = torch.tril_indices(row=cov_matrix_dim, col=cov_matrix_dim, offset=0)
        tril[:, tril_indices[0], tril_indices[1]] = lower_triangular_vector

        # lower-triangular matrix L with positive-valued diagonal entries, such that Sigma=LL^T
        # apply softplus to ensure the diagonal elements are positive, from the paper
        tril[:, torch.arange(cov_matrix_dim), torch.arange(cov_matrix_dim)] = F.softplus(tril[:, torch.arange(cov_matrix_dim), torch.arange(cov_matrix_dim)])

        dist = MultivariateNormal(mu, scale_tril=tril)
        return dist
