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
import time

# third-party imports
import gymnasium
from tensordict import TensorDict
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# local imports
from logger import create_logger
from utils import Utils

class MPOTrainer:
    
    def __init__(
        self,
        env,
        agent,
        replay_buffer,
        log_dir,
        checkpoint_dir,
        experiment_name: str = "mpo_experiment",
        episodes_total: int = 100000,
        evaluation_episodes: int = 10,
        number_episodes_per_update: int = 10,
        max_steps_per_episode: int = 1000,
        render: bool = False,
        test_only: bool = False,
        device = torch.device("cuda:0"),) -> None:
        
        self._train_env = env
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._log_dir = log_dir
        self._checkpoint_dir = checkpoint_dir
        self._episodes_total = episodes_total
        self._evaluation_episodes = evaluation_episodes
        self._number_episodes_per_update = number_episodes_per_update
        self._max_steps_per_episode = max_steps_per_episode
        self._device = device
        self._total_episodes = 0
        self._logger = create_logger(f"{experiment_name}_trainer")
        self._render = render
        self._test_only = test_only

        # tensorboard logging
        self._tb_writer = SummaryWriter(log_dir=self._log_dir)

        # if evaluate only mode, enable rendering and load the models
        if self._test_only:
            self._render = True
            self._agent.load_checkpoint(self._checkpoint_dir)

        self._evaluation_env = gymnasium.make(env.spec.id) if not self._render else gymnasium.make(env.spec.id, render_mode="human")

    def _collect_trajectories(self):

        episodic_rewards = []

        for _ in tqdm(range(self._number_episodes_per_update), desc="Trajectory collection", unit="episodes"):
            terminated = False
            state, _ = self._train_env.reset(seed=123)
            steps = 0

            states = None
            actions = None
            rewards = None
            next_states = None
            terminations = None
            
            while not terminated and steps < self._max_steps_per_episode:
                state_tensor = Utils.create_tensor(state, self._device)

                action = self._agent.generate_action(state_tensor)
                next_state, reward, terminated, _, _ = self._train_env.step(action)

                if states is None:
                    states = state_tensor
                    actions = Utils.create_tensor(action, self._device)
                    next_states = Utils.create_tensor(next_state, self._device)
                    rewards = torch.tensor([reward]).to(torch.float32).unsqueeze(0).to(self._device)
                    terminations = torch.tensor([terminated]).to(torch.int).unsqueeze(0).to(self._device)
                else:
                    states = torch.cat((states, state_tensor), 0)
                    actions = torch.cat((actions, Utils.create_tensor(action, self._device)), 0)
                    next_states = torch.cat((next_states, Utils.create_tensor(next_state, self._device)), 0)
                    rewards = torch.cat((rewards, torch.tensor([reward]).to(torch.float32).unsqueeze(0).to(self._device)), 0)
                    terminations = torch.cat((terminations, torch.tensor([terminated]).to(torch.int).unsqueeze(0).to(self._device)), 0)

                # write to replay buffer
                state = next_state
                steps += 1
            
            # tensordict of one trajectory
            sars = Utils.create_sars(states, next_states, actions, rewards, terminations, self._device)

            self._replay_buffer.extend(sars) # buffer of trajectories stored as sars tuples
            self._total_episodes += 1
            episodic_rewards.append(rewards.sum().item())

        return sum(episodic_rewards)/len(episodic_rewards)
            
    def train(self):
        
        if self._test_only:
            self._logger.warning("Test only mode. Skipping training.")
            return
        
        done = False
        start_time = time.time()    
        self._logger.info(f"Starting training for {self._episodes_total} episodes.")  

        # initialize
        start_time = time.time()   
        self._total_episodes = 0
        training_done = False
        itr = 0

        # train the agent
        while self._total_episodes < self._episodes_total and not training_done:

            mean_episodic_reward = self._collect_trajectories()
            log, training_done = self._agent.perform_updates(self._replay_buffer)
            start_time = time.time()
            itr += 1

            self._logger.info(f"Iteration took {time.time() - start_time} seconds. Total episodes: {self._total_episodes}. Iteration: {itr}.")

            # tensorboard logging
            self._tb_writer.add_scalar("Mean episodic reward", mean_episodic_reward, itr)

            for key in log:
                self._logger.info(f"{key}: {log[key]}")

                # tensorboard logging
                self._tb_writer.add_scalar(key, log[key], itr)

            self._logger.info(f"Mean episodic reward: {mean_episodic_reward}")
            self._tb_writer.flush()

            # checkpointing
            self._agent.save_checkpoint(self._checkpoint_dir)

            mean_evaluation_reward = self._evaluate(max_episodic_length=self._max_steps_per_episode)

            # tensorboard logging
            self._tb_writer.add_scalar("Mean evaluation episodic reward", mean_evaluation_reward, itr)

                    
        self._train_env.close()
        self._evaluation_env.close()
        self._tb_writer.close()
    
    def _evaluate(self, max_episodic_length=100, num_episodes=None):
        eval_rewards = []

        if num_episodes is None:
            num_episodes = self._evaluation_episodes

        for _ in range(num_episodes):
            state, _ = self._evaluation_env.reset(seed=123)

            terminated = False
            episode_reward = 0
            steps = 0

            while not terminated and steps < max_episodic_length:
                state_dict = Utils.create_tensor(state, self._device)
                action = self._agent.generate_action(state_dict, eval=True)
                next_state, reward, terminated, _, _ = self._evaluation_env.step(action)
                state = next_state
                episode_reward += reward
                steps += 1

                if self._render:
                    self._evaluation_env.render()

            eval_rewards.append(episode_reward)

        self._logger.info(f"Mean evaluation episodic rewards: {sum(eval_rewards)/len(eval_rewards)}")

        return sum(eval_rewards)/len(eval_rewards)

    
    def test(self, num_episodes=10):
        self._evaluate(max_episodic_length=self._max_steps_per_episode, num_episodes=num_episodes)
