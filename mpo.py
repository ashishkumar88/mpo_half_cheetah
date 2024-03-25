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
from copy import deepcopy

# third-party imports
import torch
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
from scipy.optimize import minimize

# local imports
from model import Actor, Critic
from utils import Utils

class MPOAgent:

    def __init__(self,
        env,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        num_updates_per_iter: int = 1000,
        epsilon_g: float = 0.2,
        epsilon_mu: float = 0.1,
        epsilon_sigma: float = 0.1,
        device = torch.device("cuda:0"),
        num_action_samples_for_expectation: int = 64,
        eta_mu_lr: float = 1.0,
        eta_sigma_lr: float = 100.0,
        eta_m_conv_steps: int = 10,
        ) -> None:
        
        self._env = env
        
        self._actor = Actor(
            env,
            actor_hidden_dim,
            device=device
        )

        self._critic = Critic(
            env,
            critic_hidden_dim,
            device=device
        )

        self._target_actor = deepcopy(self._actor)  
        self._target_critic = deepcopy(self._critic)

        self._actor_optimizer = Adam(
            self._actor.parameters(), 
            lr=actor_lr
        )

        self._critic_optimizer = Adam(
            self._critic.parameters(), 
            lr=critic_lr
        )

        self._gamma = gamma
        self._device = device
        self._num_updates_per_iter = num_updates_per_iter

        # MPO specific parameters
        self._epsilon_g = epsilon_g
        self._eta = np.random.rand()
        self._eta_mu = 1e-6
        self._eta_sigma = 1e-6 # very small etas per paper
        self._epsilon_mu = epsilon_mu
        self._epsilon_sigma = epsilon_sigma
        self._num_action_samples_for_expectation = num_action_samples_for_expectation
        self._eta_mu_lr = eta_mu_lr
        self._eta_sigma_lr = eta_sigma_lr
        self._eta_m_conv_steps = eta_m_conv_steps

    def get_config(self):
        config_as_str = ""
        for key, value in self.__dict__.items():
            config_as_str += f"{key}: {value}\n"

        return config_as_str
    
    def generate_action(self, state, eval=False):
        """ Generate an action based on the current policy
        """

        action = None
        if eval:
            self._actor.eval()
            action = self._actor.get_action(state)
            self._actor.train()
        else:
            action = self._target_actor.get_action(state) 

        return action

    # based on suggestion here https://discuss.pytorch.org/t/kl-divergence-between-two-multivariate-gaussian/53024/10
    # for trace https://discuss.pytorch.org/t/is-there-a-way-to-compute-matrix-trace-in-batch-broadcast-fashion/43866
    def _multivariate_gaussian_kl(self, mu1, sigma1, mu2, sigma2):
        """ Compute the KL divergence between two multivariate Gaussian distributions
        """

        sigma2_inv = torch.inverse(sigma2) # batch inverse
        mu_diff = mu2 - mu1
        mu_diff = mu_diff.unsqueeze(-1) 

        trace = torch.diagonal(sigma2_inv @ sigma1, dim1=-2, dim2=-1).sum(-1) # torch.trace does not work on batch
        C_mu = 0.5 * (torch.log(torch.det(sigma2) / torch.det(sigma1)) - mu1.shape[0] +  trace)
        C_mu = C_mu.mean()
        C_sigma = 0.5 * (torch.transpose(mu_diff, -1, -2) @ sigma2_inv @ mu_diff)
        C_sigma = C_sigma.squeeze(-1).squeeze(-1).mean()
        kl =  C_mu + C_sigma

        return kl, C_mu, C_sigma

    def perform_updates(self, replay_buffer):
        """ Perform the MPO updates
        """

        actor_losses = []
        critic_losses = []

        for _ in tqdm(range(self._num_updates_per_iter), desc="MPO Iteration", unit="updates"):

            batch = replay_buffer.sample().to(self._device).detach()

            # policy evaluation

            # compute the target Q values            
            with torch.no_grad():
                next_states = batch[("next", "state")]
                target_policy = self._target_actor.get_distribution(next_states)
                next_actions = target_policy.sample()

                next_q = self._target_critic.forward(next_states, next_actions)
                target_q_values = batch["reward"] + self._gamma * next_q

                # TODO implement retrace based on https://github.com/hill-a/stable-baselines/blob/3d115a3f1f5a755dc92b803a1f5499c24158fb21/stable_baselines/acer/acer_simple.py#L50-L52

            # compute the Q values
            q_values = self._critic(batch["state"], batch["action"])
            
            # compute the critic loss
            critic_loss = F.mse_loss(q_values, target_q_values.detach())
            critic_losses.append(critic_loss.item())

            # update the critic
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            self._critic_optimizer.step()

            # E-Step of Policy 
            with torch.no_grad():
                states = batch["state"]  
                behavior_policy = self._target_actor.get_distribution(states)  
                actions = behavior_policy.sample((self._num_action_samples_for_expectation,))  
                expanded_states = states[None, ...].expand(self._num_action_samples_for_expectation, -1, -1)
                target_q = self._target_critic.forward(expanded_states.reshape(-1, self._env.observation_space.shape[0]), actions.reshape(-1, self._env.action_space.shape[0])).reshape(self._num_action_samples_for_expectation, states.shape[0])
                target_q_nd = target_q.cpu().transpose(0, 1).numpy()  

            g = lambda eta : eta * self._epsilon_g + eta * np.mean(np.log(np.mean(np.exp(target_q_nd / eta), axis=1))) # convex function to minimize
            bnds = [(0, None)] # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            res = minimize(g, np.array([self._eta]), method='SLSQP', bounds=bnds) # https://stackoverflow.com/questions/43648073/what-is-the-fastest-way-to-minimize-a-function-in-python
            self._eta = res.x[0]

            q_i = torch.softmax(target_q / self._eta, dim=0)
            
            # M-Step of Policy 
            for _ in range(self._eta_m_conv_steps):
                policy = self._actor.get_distribution(states)
                loss_p = torch.mean( q_i * policy.expand((self._num_action_samples_for_expectation, states.shape[0])).log_prob(actions))
                _, C_mu, C_sigma  = self._multivariate_gaussian_kl(behavior_policy.loc, behavior_policy.covariance_matrix, policy.loc, policy.covariance_matrix)

                self._eta_mu -= self._eta_mu_lr * (self._epsilon_mu - C_mu).detach().item()
                self._eta_sigma -= self._eta_sigma_lr * (self._epsilon_sigma - C_sigma).detach().item()

                self._eta_mu = 0.0 if self._eta_mu < 0.0 else self._eta_mu
                self._eta_sigma = 0.0 if self._eta_sigma < 0.0 else self._eta_sigma

                loss_l = - (loss_p + self._eta_mu * (self._epsilon_mu - C_mu) + self._eta_sigma * (self._epsilon_sigma - C_sigma))
                
                self._actor_optimizer.zero_grad()
                loss_l.backward()
                self._actor_optimizer.step()

                actor_losses.append(loss_l.item())

        # update the target networks
        self._target_actor.load_state_dict(self._actor.state_dict())
        self._target_critic.load_state_dict(self._critic.state_dict())

        return {"critic_loss" : sum(critic_losses)/len(critic_losses), "actor_loss" : sum(actor_losses) / len(actor_losses)}, False

    def save_checkpoint(self, checkpoint_dir):
        """ Save the agent's checkpoint
        """

        torch.save(self._actor.state_dict(), f"{checkpoint_dir}/actor.pth")
        torch.save(self._critic.state_dict(), f"{checkpoint_dir}/critic.pth")   

    def load_checkpoint(self, checkpoint_dir):
        """ Load the agent's checkpoint
        """

        self._actor.load_state_dict(torch.load(f"{checkpoint_dir}/actor.pth"))
        self._critic.load_state_dict(torch.load(f"{checkpoint_dir}/critic.pth"))