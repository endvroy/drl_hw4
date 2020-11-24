from mpc import *

if __name__ == '__main__':
    env_model = EnvModel(24, 4, [100, 200, 100])
    policy = Policy(24, [100, 200, 100], 4)
    mpc = MPC(env_model=env_model,
              policy=policy,
              rollout_steps=10,
              buffer_capacity=20000,
              batch_size=128,
              model_lr=5e-4,
              policy_lr=1e-4,
              device=torch.device('cuda'),
              tb_path='mpc_logs',
              ckpt_path='mpc_logs')
    mpc.load_checkpoint('mpc_logs/5000.pt')
    mpc.eval()
