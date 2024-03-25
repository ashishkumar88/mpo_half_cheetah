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

# third-party imports
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import numpy as np

# local imports
from utils import Utils

class Actor(nn.Module):
    def __init__(
        self,
        env,
        hidden_dims: int = 256,
        device = torch.device("cuda:0"),
        activation = nn.ReLU(),
        ) -> None:
        
        super(Actor, self).__init__()

        self._device = device
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._env = env
        self._action_low = torch.from_numpy(self._env.action_space.low)[None, ...].to(self._device)  
        self._action_high = torch.from_numpy(self._env.action_space.high)[None, ...].to(self._device) 
        
        self._hiddens = nn.Sequential(
            nn.Linear(self._state_dim, hidden_dims),
            activation,
            nn.Linear(hidden_dims, hidden_dims),
            activation,
        )

        self._mu_layer = nn.Linear(hidden_dims, self._action_dim)
        self._lower_triangular_layer = nn.Linear(hidden_dims, self._action_dim * (self._action_dim + 1) // 2) # The paper assumes a full covariance matrix

        self.to(self._device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self._hiddens(state) 
        mu = torch.sigmoid(self._mu_layer(h)) # squish mu between 0 and 1
        mu = self._action_low + (self._action_high - self._action_low) * mu
        lower_triangular_vector = self._lower_triangular_layer(h) # less parameters than covariance matrix
        return mu, lower_triangular_vector
    

    def get_action(self, state: torch.Tensor):
        with torch.no_grad():
            mu, lower_triangular_vector = self.forward(state)
            dist = Utils.convert_params_to_multivariate_normal(mu, lower_triangular_vector, self._action_dim)
            action = dist.sample()
            action = action.squeeze().cpu().numpy()
            return action
        
    def get_distribution(self, state: torch.Tensor):
        mu, lower_triangular_vector = self.forward(state)

        # the paper assumes a full covariance matrix, so not using the torch.distributions.Normal
        # instead, we will use the MultivariateNormal distribution which can use a lower triangular matrix
        # section D in the appendix
        dist = Utils.convert_params_to_multivariate_normal(mu, lower_triangular_vector, self._action_dim) 
        return dist
    
class Critic(nn.Module):
    def __init__(
        self,
        env,
        hidden_dims: int = 256,
        device = torch.device("cuda:0"),
        activation = nn.ReLU(),
        ) -> None:
        
        super(Critic, self).__init__()
        
        self._model = nn.Sequential(
            nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], hidden_dims),
            activation,
            nn.Linear(hidden_dims, hidden_dims),
            activation,
            nn.Linear(hidden_dims, 1)
        )

        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._model(torch.cat([state, action], dim=-1))
    
## END OF FILE