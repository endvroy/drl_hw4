import pytorch_lightning as pl
from behavior_cloning import BipedalModule
from pytorch_lightning.callbacks import ModelCheckpoint


def train():
    bipedal_module = BipedalModule(data_path='data_small_stacked.pkl',
                                   recording_path='videos',
                                   window_size=1,
                                   batch_size=64,
                                   lr=1e-3,
                                   )
    checkpoint_callback = ModelCheckpoint(save_top_k=-1)
    trainer = pl.Trainer(gpus=1,
                         default_root_dir='logs/bipedal_stacked',
                         early_stop_callback=False,
                         checkpoint_callback=checkpoint_callback,
                         max_epochs=8,
                         )
    trainer.fit(bipedal_module)


if __name__ == '__main__':
    train()
