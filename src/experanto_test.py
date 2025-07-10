import numpy as np
import torch
from experanto.configs import DEFAULT_CONFIG
from experanto.datasets import OdorDataset
from experanto.experiment import Experiment
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

base_path = "/user/burghardt17/u17926/.project/dir.project/rene/data/experanto/"
paths = [
    "030063__24_05_16",
]
full_paths = [base_path + path + "/" for path in paths]


cfg = DEFAULT_CONFIG.copy()

with open_dict(cfg):
    cfg.dataset = {
        "cache_data": False,
    }
    cfg.dataset.modality_config = {"odor_trials": {}, "responses": {}}
    cfg.dataset.modality_config.odor_trials.interpolation = {
        "interpolation_mode": "nearest_neighbor"
    }
    cfg.dataset.modality_config.responses.interpolation = {
        "interpolation_window": 300,
        "interpolation_align": "left",
    }

    cfg.dataloader.num_workers = 0
    cfg.dataloader.prefetch_factor = None
    cfg.dataloader.batch_size = 16
    cfg.dataloader.pin_memory = False
    cfg.dataloader.shuffle = False
    cfg.dataloader.drop_last = False

print(OmegaConf.to_yaml(cfg))

experiment = Experiment(base_path + paths[0], cfg.dataset.modality_config)
interp_data, _ = experiment.interpolate(np.array([70.0]))

train_dl = DataLoader(
    dataset=OdorDataset(full_paths[0], **cfg.dataset), **cfg.dataloader
)

batch = next(iter(train_dl))

breakpoint()
