from behavior_cloning import BipedalModule

if __name__ == '__main__':
    ckpt_path = 'logs/bipedal_stacked/lightning_logs/version_0/checkpoints/epoch=7.ckpt'
    bipedal_module = BipedalModule.load_from_checkpoint(ckpt_path)
    bipedal_module.run_episode(
        # recording_path='videos/example_2.mp4'
    )
