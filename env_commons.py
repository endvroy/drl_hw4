import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


def obs_to_tensor(obs):
    return torch.from_numpy(obs.astype(np.float)).float().unsqueeze(0)


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


class BipedalDataset(Dataset):
    def __init__(self, fpath):
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        self.data = []
        for x in data:
            self.data.append(
                dict(
                    state=obs_to_tensor(x[0]).squeeze(0),
                    action=action_to_tensor(x[1]).squeeze(0),
                    next_state=obs_to_tensor(x[2]).squeeze(0),
                    reward=scalar_to_tensor(x[3]).squeeze(0),
                    terminated=scalar_to_tensor(x[4]).squeeze(0)
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
