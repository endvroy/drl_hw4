import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
import math
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os
import json
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from env_commons import *


def build_nn(inp_dim, hidden_dims, out_dim):
    dims = [inp_dim] + hidden_dims + [out_dim]
    nets = []
    for i in range(len(dims) - 1):
        nets.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            nets.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*nets)


class StackedPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super().__init__()
        self.net = build_nn(4 * state_dim, hidden_dims, action_dim)
        self.action_dim = action_dim

    def forward(self, state):
        inp = state.reshape([state.shape[0], -1])
        out = self.net(inp)
        actions = torch.tanh(out)
        return actions

    def pick_action(self, state):
        return self.forward(state)


class BipedalModule(pl.LightningModule):
    def __init__(self,
                 data_path,
                 recording_path,
                 window_size,
                 batch_size,
                 lr,
                 data_points=None
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.data_points = data_points
        self.dataset = None
        self.window_size = window_size
        self.policy = StackedPolicy(24, [50, 50], 4)
        self.example_input_array = torch.empty([1, 4, 24], dtype=torch.float)
        self.recording_path = recording_path
        os.makedirs(self.recording_path, exist_ok=True)

        self.batch_size = batch_size
        self.lr = lr

    def load_and_split_dataset(self):
        if self.dataset is None:
            if self.data_points is None:
                self.dataset = BipedalDataset(self.data_path)
            else:
                self.dataset = Subset(BipedalDataset(self.data_path),
                                      range(self.data_points))
            val_cnt = int(0.1 * len(self.dataset))
            self.train_set, self.val_set = torch.utils.data.random_split(self.dataset,
                                                                         [len(self.dataset) - val_cnt, val_cnt],
                                                                         generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        self.load_and_split_dataset()
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=6)

    def val_dataloader(self):
        self.load_and_split_dataset()
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=6)

    def configure_optimizers(self):
        return torch.optim.Adam(self.policy.parameters(),
                                lr=self.lr)

    def forward(self, x):
        return self.policy(x)

    def training_step(self, batch, batch_nb):
        state = batch['state']
        action = batch['action']
        pred_action = self.forward(state)
        loss = F.mse_loss(pred_action, action)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_nb):
        state = batch['state']
        action = batch['action']
        pred_action = self.forward(state)
        loss = F.mse_loss(pred_action, action)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def on_epoch_end(self):
        return
        recording_path = f'{self.recording_path}/epoch_{self.current_epoch}.mp4'
        reward = self.run_episode(recording_path=recording_path)

    def random_capture(self, max_timesteps=1000):
        env = gym.make('CarRacing-v0').unwrapped
        state = env.reset()
        # maintain a window of history
        window = deque(maxlen=4)
        # fill in initial value of blank
        for i in range(4):
            window.append(torch.zeros((1, 24)))

        captured = []

        for step in range(max_timesteps):
            # randomly capture
            if torch.rand([1]) < 0.01 and len(captured) < 6:
                captured.append(state)
            state = obs_to_tensor(state).to(self.device)
            window.append(state)
            obs = torch.stack(list(window)).unsqueeze(0)
            with torch.no_grad():
                action_tensor = self.policy(obs)
            action = action_tensor_to_action(action_tensor)
            next_state, r, done, info = env.step(action)
            state = next_state
            step += 1

            if done or step > max_timesteps or len(captured) >= 6:
                break

        for i, img in enumerate(captured):
            plt.imshow(img)
            plt.savefig(f'figs/clone_{i}.png')

        env.close()

    def run_episode(self, rendering=True, recording_path=None):
        env = gym.make('BipedalWalkerHardcore-v3')
        if recording_path is not None:
            video_recorder = VideoRecorder(env, recording_path)

        episode_reward = 0

        state = env.reset()
        # maintain a window of history
        window = deque(maxlen=4)
        # fill in initial value of blank
        for i in range(4):
            window.append(torch.zeros((1, 24)))

        while True:
            state = obs_to_tensor(state).to(self.device)
            window.append(state)
            obs = torch.stack(list(window)).unsqueeze(0)
            with torch.no_grad():
                action_tensor = self.policy(obs)
            action = action_tensor_to_action(action_tensor)
            next_state, r, done, info = env.step(action)
            episode_reward += r
            state = next_state

            if rendering:
                env.render()

            if recording_path is not None:
                video_recorder.capture_frame()

            if done:
                break

        env.close()
        if recording_path is not None:
            video_recorder.close()
        return episode_reward

    def final_test(self, max_timesteps=1600):
        rendering = True

        n_test_episodes = 15  # number of episodes to test

        episode_rewards = []
        for i in range(n_test_episodes):
            episode_reward = self.run_episode(rendering=rendering, max_timesteps=max_timesteps)
            episode_rewards.append(episode_reward)
