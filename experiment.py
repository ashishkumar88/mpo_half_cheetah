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
import argparse
import os
import datetime

# third-party imports
import gymnasium
from torchrl.data.replay_buffers import ReplayBuffer, LazyMemmapStorage
import torch

# local imports
from utils import Utils
from mpo import MPOAgent
from trainer import MPOTrainer

class ContinousActionMPOEXperiment:
    def __init__(
        self,
        config_path: str = None,
        test_only: bool = False,
        checkpoint_dir: str = None) -> None:

        if config_path is not None and not os.path.isabs(config_path) and os.path.isfile(config_path):
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config/mujoco_half_cheetah.yaml")
        elif config_path is None or not os.path.isfile(config_path):
            raise ValueError("The configuration file path is not valid.")
        
        if test_only and checkpoint_dir is None:
            raise ValueError("The checkpoint directory should be provided in the test only mode.")
        
        Utils.load_config(config_path, self) # load configs from the yaml file as attributes of the class
        
        # create the environment
        self._env = gymnasium.make(self._env_name)

        # log and checkpoint configuration
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if not hasattr(self, '_base_dir') or self._base_dir is None:
            self._base_dir = os.path.dirname(os.path.realpath(__file__))

        if not os.path.exists(os.path.join(self._base_dir, "logs")):
            os.makedirs(os.path.join(self._base_dir, "logs"))
        
        if not os.path.exists(os.path.join(self._base_dir, "logs", self._env_name, timestamp)):
            os.makedirs(os.path.join(self._base_dir, "logs", self._env_name, timestamp))

        if not os.path.exists(os.path.join(self._base_dir, "checkpoints")):
            os.makedirs(os.path.join(self._base_dir, "checkpoints"))
        
        if not os.path.exists(os.path.join(self._base_dir, "checkpoints", self._env_name, timestamp)):
            os.makedirs(os.path.join(self._base_dir, "checkpoints", self._env_name, timestamp))

        self._log_dir = os.path.join(self._base_dir, "logs", self._env_name, timestamp)
        self._checkpoint_dir = checkpoint_dir if checkpoint_dir is not None and test_only else os.path.join(self._base_dir, "checkpoints", self._env_name, timestamp) # checkpoint dir should be provided if test_only is True

        # training device
        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 

        # agent configs
        self._agent = MPOAgent(
            env = self._env,
            actor_hidden_dim = self._actor_hidden_dim if hasattr(self, '_actor_hidden_dim') else 256,
            critic_hidden_dim =  self._critic_hidden_dim if hasattr(self, '_critic_hidden_dim') else 256,
            actor_lr = self._actor_lr if hasattr(self, '_actor_lr') else 1e-3,
            critic_lr = self._critic_lr if hasattr(self, '_critic_lr') else 1e-3,
            gamma = self._gamma if hasattr(self, '_gamma') else 0.99,
            num_updates_per_iter = self._num_updates_per_iter if hasattr(self, '_num_updates_per_iter') else 1000,
            epsilon_g = self._epsilon_g if hasattr(self, '_epsilon_g') else 0.2,
            epsilon_mu = self._epsilon_mu if hasattr(self, '_epsilon_mu') else 0.1,
            epsilon_sigma = self._epsilon_sigma if hasattr(self, '_epsilon_sigma') else 0.1,
            num_action_samples_for_expectation = self._num_action_samples_for_expectation if hasattr(self, '_num_action_samples_for_expectation') else 64,
            eta_mu_lr=self._eta_mu_lr if hasattr(self, '_eta_mu_lr') else 1e-3,
            eta_sigma_lr=self._eta_sigma_lr if hasattr(self, '_eta_sigma_lr') else 1e-3,
            device = self._device,            
        )
            
        # replay buffer
        self._replay_buffer_size = 1e5 if not hasattr(self, '_replay_buffer_size') else self._replay_buffer_size
        self._batch_size = 256 if not hasattr(self, '_batch_size') else self._batch_size
        self._replay_buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._replay_buffer_size), batch_size=self._batch_size,)

        # trainer
        self._trainer = MPOTrainer(
            env = self._env,
            agent = self._agent,
            replay_buffer = self._replay_buffer,
            log_dir = self._log_dir,
            checkpoint_dir = self._checkpoint_dir,
            experiment_name = self._experiment_name if hasattr(self, '_experiment_name') else "mpo_half_cheetah_experiment", 
            episodes_total = self._episodes_total if hasattr(self, '_episodes_total') else 100000,
            evaluation_episodes = self._evaluation_episodes if hasattr(self, '_evaluation_episodes') else 10,
            number_episodes_per_update = self._number_episodes_per_update if hasattr(self, '_number_episodes_per_update') else 10,
            max_steps_per_episode = self._max_steps_per_episode if hasattr(self, '_max_steps_per_episode') else 1000,
            render = self._render if hasattr(self, '_render') else False,
            test_only = test_only,
            device=self._device
        )

        # create a meta log entry
        self._meta_log_file = os.path.join(self._base_dir, "logs", self._env_name, "experiments.txt")

        with open(self._meta_log_file, "a") as f:
            f.write(f"Experiment started at {timestamp}\n")
            f.write(f"Environment: {self._env_name}\n")
            f.write(f"Agent: {self._agent.get_config()}\n")
            f.write("\n")

        self._test_only = test_only

    def run(self) -> None:
        if not self._test_only:
            self._trainer.train()

        self._trainer.test()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the experiment for the MPO algorithm")
    argparser.add_argument("--config", default=None, help="The path to the configuration file")
    argparser.add_argument("--test", help="Test only mode", action="store_true")
    argparser.add_argument("--checkpoint_dir", default=None, help="The path to the checkpoint directory")
    args = argparser.parse_args()

    experiment = ContinousActionMPOEXperiment(args.config, args.test, args.checkpoint_dir)
    experiment.run()
