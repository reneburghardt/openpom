import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToTensor

from experanto.configs import DEFAULT_MODALITY_CONFIG
from experanto.experiment import Experiment


class OdorDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        cache_data: bool = False,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
        summation_window: int = 300,
    ) -> None:
        self.root_folder = Path(root_folder)
        self.modality_config = instantiate(modality_config)

        self._experiment = Experiment(
            root_folder,
            modality_config,
            cache_data=cache_data,
        )
        self.device_names = self._experiment.device_names
        self.transforms = self.initialize_transforms()

        self._valid_odor_times = self._get_valid_times_from_trials()

    def initialize_transforms(self):
        """
        Initializes the transforms for each device based on the modality config.
        :return:
        """
        transforms = {}
        for device_name in self.device_names:
            transform_list = []
            if 'transforms' in self.modality_config[device_name]:
                for v in self.modality_config[device_name].transforms.values():
                    transform_list.append(v)

            if device_name != "odor_trials":
                transform_list.append(lambda x: torch.from_numpy(x).float())

            transforms[device_name] = Compose(transform_list)
        return transforms

    def _get_valid_times_from_trials(self) -> np.ndarray:
        """
        Extracts valid odor times from the trials of the experiment.
        Returns:
            np.ndarray: Array of valid odor times.
        """
        if "valid_condition" not in self.modality_config["odor_trials"]:
            return self._experiment.devices["odor_trials"].timestamps[:-1]
        
        valid_conditions = self.modality_config["odor_trials"]["valid_condition"]
        valid_conditions = valid_conditions if isinstance(valid_conditions, list) else [valid_conditions]
        valid_times = []
        for time, trial in zip(self._experiment.devices["odor_trials"].timestamps[:-1], self._experiment.devices["odor_trials"].trials):
            for valid_condition in valid_conditions:
                and_valid = True
                for k, v in valid_condition.items():
                    if k not in trial or trial[k] != v:
                        and_valid = False
                        break

                if and_valid:
                    valid_times.append(time)
                    break
        return np.array(valid_times)

    def __len__(self):
        return len(self._valid_odor_times)

    def __getitem__(self, idx) -> dict:
        out = {}
        s = np.array([self._valid_odor_times[idx]])
        for device_name in self.device_names:
            data, _ = self._experiment.interpolate(s, device=device_name)
            out[device_name] = self.transforms[device_name](data)

        return out