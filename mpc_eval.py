from mpc import *

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
              buffer_capacity=1000,
              batch_size=128,
              model_lr=1e-3,
              policy_lr=1e-4,
              device=torch.device('cuda'),
              tb_path='mpc_logs',
              ckpt_path='mpc_logs')
    mpc.load_checkpoint('mpc_logs/5000.pt')
    mpc.eval()
