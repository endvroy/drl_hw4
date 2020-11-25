import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from env_commons import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def build_nn(inp_dim, hidden_dims, out_dim):
    dims = [inp_dim] + hidden_dims + [out_dim]
    nets = []
    for i in range(len(dims) - 1):
        nets.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            nets.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*nets)


class EnvModel(nn.Module):
    def __init__(self, state_dim, action_dim,
                 dyn_hidden_dims,
                 fail_hidden_dims,
                 ):
        super().__init__()
        self.dyn_net = build_nn(state_dim + action_dim, dyn_hidden_dims, state_dim + 1)
        self.fail_net = build_nn(state_dim, fail_hidden_dims, 1)
        self.state_dim = state_dim

    def forward(self, state, action):
        inp = torch.cat([state, action], dim=1)
        dyn_out = self.dyn_net(inp)
        next_state, other_reward = dyn_out.split([self.state_dim, 1], dim=1)
        fail = torch.sigmoid(self.fail_net(next_state)).squeeze(1)
        other_reward = other_reward.squeeze(1)
        return next_state, other_reward, fail


class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super().__init__()
        self.net = build_nn(state_dim, hidden_dims, action_dim)
        self.action_dim = action_dim

    def forward(self, state):
        out = self.net(state)
        actions = torch.tanh(out)
        return actions

    def pick_action(self, state):
        return self.forward(state)


class TransitionBuffer:
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)

    def add(self, transition):
        self.mem.append(transition)

    def sample(self, batch_size):
        return cat_transitions(random.choices(self.mem, k=batch_size))

    def __len__(self):
        return len(self.mem)

    def __iter__(self):
        return iter(self.mem)


def obs_to_tensor(obs):
    return torch.from_numpy(obs).float().unsqueeze(0)


def action_to_tensor(action):
    return torch.from_numpy(action).float().unsqueeze(0)


def scalar_to_tensor(scalar):
    return torch.Tensor([scalar])


def transition(state, action, next_state, reward, terminated):
    return TransitionBatch(
        state=obs_to_tensor(state),
        action=action_to_tensor(action),
        next_state=obs_to_tensor(next_state),
        reward=scalar_to_tensor(reward),
        terminated=scalar_to_tensor(terminated)
    )


class MPC:
    def __init__(self,
                 env_model,
                 policy,
                 rollout_steps,
                 buffer_capacity,
                 pretrain_batch_size,
                 batch_size,
                 model_lr,
                 policy_lr,
                 device,
                 dataset_path,
                 tb_path,
                 ckpt_path):
        self.device = device
        self.env_model = env_model.to(self.device)
        self.policy = policy.to(self.device)
        self.rollout_steps = rollout_steps
        self.buffer_capacity = buffer_capacity
        self.buffer = TransitionBuffer(capacity=self.buffer_capacity)
        self.pretrain_batch_size = pretrain_batch_size
        self.batch_size = batch_size
        self.model_lr = model_lr
        self.policy_lr = policy_lr
        self.model_optim = torch.optim.Adam(self.env_model.parameters(), lr=self.model_lr)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.dataset_path = dataset_path
        os.makedirs(tb_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        self.tb_path = tb_path
        self.writer = SummaryWriter(self.tb_path)
        self.ckpt_path = ckpt_path
        self.global_step = 0

    def config_dataloader(self):
        self.dataset = BipedalDataset(self.dataset_path)
        self.dataloader = DataLoader(self.dataset, batch_size=self.pretrain_batch_size, shuffle=True)

    def pretrain_env_model(self):
        self.config_dataloader()
        it = iter(self.dataloader)
        for step in range(1000):
            batch = next(it)
            trans_batch = TransitionBatch(**batch).to(self.device)
            state_loss, other_reward_loss, fail_loss, model_loss = self.train_env_model_step(trans_batch)
            self.writer.add_scalar('pretrain_state_loss', state_loss)
            self.writer.add_scalar('pretrain_other_reward_loss', other_reward_loss)
            self.writer.add_scalar('pretrain_fail_loss', fail_loss)
            self.writer.add_scalar('pretrain_model_loss', model_loss)

    def train_env_model_step(self, batch):
        self.model_optim.zero_grad()
        states, actions, next_states, rewards, terminated = batch.to(self.device).unpack()
        fails = terminated
        other_rewards = rewards + 100 * fails
        predicted_states, predicted_other_rewards, predicted_fail = self.env_model(states, actions)
        state_loss = F.mse_loss(predicted_states, next_states)
        other_reward_loss = F.mse_loss(predicted_other_rewards, other_rewards)
        fail_loss = F.binary_cross_entropy(predicted_fail, fails)
        model_loss = state_loss + other_reward_loss + fail_loss
        model_loss.backward()
        self.model_optim.step()
        return state_loss, other_reward_loss, fail_loss, model_loss

    def train_step(self, env, state):
        state_tensor = obs_to_tensor(state).to(self.device)
        # train env module
        if self.global_step > self.batch_size:
            batch = self.buffer.sample(self.batch_size)
            state_loss, other_reward_loss, fail_loss, model_loss = self.train_env_model_step(batch)
            self.writer.add_scalar('state_loss', state_loss, self.global_step)
            self.writer.add_scalar('other_reward_loss', other_reward_loss, self.global_step)
            self.writer.add_scalar('fail_loss', fail_loss, self.global_step)
            self.writer.add_scalar('model_loss', model_loss, self.global_step)

        # train policy module
        # rollout
        total_reward = 0
        rollout_state = state_tensor
        for i in range(self.rollout_steps):
            action_tensor = self.policy.pick_action(rollout_state)
            next_state, other_reward, fail = self.env_model(rollout_state, action_tensor)
            reward = other_reward - 100 * fail
            total_reward += reward
            rollout_state = next_state

        self.policy_optim.zero_grad()
        neg_reward = -total_reward
        neg_reward.backward()
        self.policy_optim.step()
        self.writer.add_scalar('rollout_reward', total_reward, self.global_step)

        # take actual step
        with torch.no_grad():
            actual_action = self.policy.pick_action(state_tensor).cpu().squeeze(0).numpy()
        actual_next_state, actual_reward, terminated, *_ = env.step(actual_action)
        trans = transition(state=state,
                           action=actual_action,
                           next_state=actual_next_state,
                           reward=actual_reward,
                           terminated=terminated)
        self.buffer.add(trans)
        self.global_step += 1
        return actual_next_state, terminated

    def train(self):
        # self.pretrain_env_model()

        env = gym.make('BipedalWalkerHardcore-v3')
        state = env.reset()
        for i in range(5000):
            next_state, terminated = self.train_step(env, state)
            if terminated:
                state = env.reset()
            else:
                state = next_state
            if self.global_step % 500 == 0:
                print(f'{self.global_step} steps done')
        self.save_checkpoint()

    def eval(self):
        env = gym.make('BipedalWalkerHardcore-v3')
        state = env.reset()
        total_reward = 0
        while True:
            state_tensor = obs_to_tensor(state).to(self.device)
            with torch.no_grad():
                action = self.policy.pick_action(state_tensor).cpu().squeeze(0).numpy()
            next_state, reward, terminated, *_ = env.step(action)
            total_reward += reward
            env.render()
            if terminated:
                break
            else:
                state = next_state
        print(total_reward)

    def random_capture(self):
        captured = []

        env = gym.make('BipedalWalkerHardcore-v3')
        state = env.reset()
        while True:
            state_tensor = obs_to_tensor(state).to(self.device)
            with torch.no_grad():
                action = self.policy.pick_action(state_tensor).cpu().squeeze(0).numpy()
            next_state, reward, terminated, *_ = env.step(action)
            if torch.rand([1]) < 0.1 and len(captured) < 6:
                captured.append(env.render(mode='rgb_array'))
            env.render()
            if terminated or len(captured) >= 6:
                break
            else:
                state = next_state

        env.close()

        for i, img in enumerate(captured):
            plt.imshow(img)
            plt.savefig(f'dqn_{i}.png')

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.env_model.state_dict(),
            'policy_state_dict': self.policy.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.ckpt_path, f'{self.global_step}.pt'))

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.env_model.load_state_dict(checkpoint['model_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])


if __name__ == '__main__':
    env_model = EnvModel(24, 4,
                         dyn_hidden_dims=[50, 50],
                         fail_hidden_dims=[50, 50])
    policy = Policy(24, [50, 50], 4)
    mpc = MPC(env_model=env_model,
              policy=policy,
              rollout_steps=20,
              buffer_capacity=1000,
              pretrain_batch_size=256,
              batch_size=128,
              model_lr=1e-3,
              policy_lr=1e-4,
              device=torch.device('cuda'),
              dataset_path='data.pkl',
              tb_path='mpc_logs',
              ckpt_path='mpc_logs')
    mpc.train()
