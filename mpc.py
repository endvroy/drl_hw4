import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from operator import itemgetter
from torch.utils.tensorboard import SummaryWriter


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


class TransitionBatch:
    def __init__(self, state, action, next_state, reward, terminated):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.terminated = terminated

    def __len__(self):
        return self.state.shape[0]

    def to(self, device):
        return TransitionBatch(
            state=self.state.to(device),
            action=self.action.to(device),
            next_state=self.next_state.to(device),
            reward=self.reward.to(device),
            terminated=self.terminated.to(device)
        )

    def unpack(self):
        return self.state, self.action, self.next_state, self.reward, self.terminated


def cat_transitions(transitions):
    return TransitionBatch(
        state=torch.cat([transition.state for transition in transitions]),
        action=torch.cat([transition.action for transition in transitions]),
        next_state=torch.cat([transition.next_state for transition in transitions]),
        reward=torch.cat([transition.reward for transition in transitions]),
        terminated=torch.cat([transition.terminated for transition in transitions]),
    )


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


def action_tensor_to_action(action_tensor):
    return action_tensor.squeeze(0).cpu().numpy()


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
                 action_dim,
                 n_rollout_traj,
                 env_model,
                 policy,
                 rollout_steps,
                 buffer_capacity,
                 batch_size,
                 model_lr,
                 policy_lr,
                 device,
                 tb_path,
                 ckpt_path):
        self.device = device
        self.action_dim = action_dim
        self.n_rollout_traj = n_rollout_traj
        self.env_model = env_model.to(self.device)
        self.policy = policy.to(self.device)
        self.rollout_steps = rollout_steps
        self.buffer_capacity = buffer_capacity
        self.buffer = TransitionBuffer(capacity=self.buffer_capacity)
        self.batch_size = batch_size
        self.model_lr = model_lr
        self.policy_lr = policy_lr
        self.model_optim = torch.optim.Adam(self.env_model.parameters(), lr=self.model_lr)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        os.makedirs(tb_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        self.tb_path = tb_path
        self.writer = SummaryWriter(self.tb_path)
        self.ckpt_path = ckpt_path
        self.global_step = 0

    def rollout_traj(self, state):
        total_reward = 0
        actions = 2 * torch.rand((self.rollout_steps, 1, self.action_dim), device=self.device) - 1
        for i in range(self.rollout_steps):
            with torch.no_grad():
                next_state, other_reward, fail = self.env_model(state, actions[i])
                state = next_state
                total_reward += other_reward
                if fail > 0.7:
                    total_reward = -1000
                    break
        return actions[0], total_reward

    def train_step(self, env, state):
        state_tensor = obs_to_tensor(state).to(self.device)
        # train env module
        if self.global_step > self.batch_size:
            self.model_optim.zero_grad()
            batch = self.buffer.sample(self.batch_size)
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
            self.writer.add_scalar('state_loss', state_loss, self.global_step)
            self.writer.add_scalar('other_reward_loss', other_reward_loss, self.global_step)
            self.writer.add_scalar('fail_loss', fail_loss, self.global_step)
            self.writer.add_scalar('model_loss', model_loss, self.global_step)

        # random rollout
        trajs = []
        for i in range(self.n_rollout_traj):
            traj = self.rollout_traj(state_tensor)
            trajs.append(traj)

        # find best step
        best_action = max(trajs, key=itemgetter(1))[0]

        # take actual step
        with torch.no_grad():
            actual_action = action_tensor_to_action(best_action)
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
            # random rollout
            trajs = []
            for i in range(self.n_rollout_traj):
                traj = self.rollout_traj(state_tensor)
                trajs.append(traj)

            # find best step
            best_action = max(trajs, key=itemgetter(1))[0]

            next_state, reward, terminated, *_ = env.step(action_tensor_to_action(best_action))
            total_reward += reward
            env.render()
            if terminated:
                break
            else:
                state = next_state
        print(total_reward)

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
              action_dim=4,
              n_rollout_traj=50,
              policy=policy,
              rollout_steps=20,
              buffer_capacity=2000,
              batch_size=128,
              model_lr=1e-3,
              policy_lr=1e-4,
              device=torch.device('cuda'),
              tb_path='mpc_logs',
              ckpt_path='mpc_logs')
    mpc.train()
