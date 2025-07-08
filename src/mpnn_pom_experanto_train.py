from datetime import datetime

import deepchem as dc
import dgl
import torch
import numpy as np
import os
import wandb

from experanto.configs import DEFAULT_CONFIG
from includes.datasets import OdorDataset
from omegaconf import OmegaConf, open_dict
from openpom.feat.graph_featurizer import GraphConvConstants, GraphFeaturizer
from openpom.models.mpnn_pom import MPNNPOM
from openpom.utils.data_utils import get_class_imbalance_ratio
from torch.utils.data import DataLoader

base_path = "/user/burghardt17/u17926/.project/dir.project/rene/data/experanto/"
paths = [
    "030063__24_05_16",
]
full_paths = [base_path + path + "/" for path in paths]


class MPNNPOMWrapper(MPNNPOM):
    def __init__(self, *args, **kwargs):
        """
        Wrapper for MPNNPOM that applies a nonlinearity function to the output and adds an offset.

        Args:
            *args: Positional arguments for MPNNPOM.
            nonlinearity_fn (callable): Nonlinearity function to apply to the output. Defaults to ReLU.
            offset (float): Offset to add to the output before applying the nonlinearity. Defaults to 0.0.
            **kwargs: Keyword arguments for MPNNPOM.
        """
        super().__init__(*args, **kwargs)
        self.nonlinearity = torch.nn.ELU()

    def forward(self, inputs):
        outputs = super().forward(inputs)
        return self.nonlinearity(outputs) + 1


class PoissonLoss(torch.nn.Module):
    def __init__(self, bias=1e-08):
        """
        Computes Poisson loss between the output and target. Loss is evaluated by computing log likelihood that
        output prescribes the mean of the Poisson distribution and target is a sample from the distribution.

        Args:
        bias (float, optional): Value used to numerically stabilize evalution of the log-likelihood. This value is effecitvely added to the output during evaluation. Defaults to 1e-08.
        per_neuron (bool, optional): If set to True, the average/total Poisson loss is returned for each entry of the last dimension (assumed to be enumeration neurons) separately. Defaults to False.
        avg (bool, optional): If set to True, return mean loss. Otherwise returns the sum of loss. Defaults to True.
        full_loss (bool, optional): If set to True, compute the full loss, i.e. with Stirling correction term (not needed for optimization but needed for reporting of performance). Defaults to False.
        """
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        target = target.detach()
        rate = output
        loss = torch.nn.PoissonNLLLoss(
            log_input=False, eps=self.bias, reduction="none"
        )(rate, target)

        assert not (
            torch.isnan(loss).any() or torch.isinf(loss).any()
        ), "None or inf value encountered!"
        loss = loss.sum()
        return loss


class PearsonCorrelation(torch.nn.Module):
    def __init__(self, dim=-1, eps=1e-8):
        """
        Computes Pearson correlation coefficient between the output and target.

        Args:
              dim (int): Dimension along which to compute correlation (usually neuron/channel dimension).
              eps (float, optional): Value used to numerically stabilize evalution of the standard deviation. Defaults to 1e-8.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, output, target, **kwargs):
        y1 = (output - output.mean(dim=self.dim, keepdim=True)) / (
            output.std(dim=self.dim, keepdim=True) + self.eps
        )
        y2 = (target - target.mean(dim=self.dim, keepdim=True)) / (
            target.std(dim=self.dim, keepdim=True) + self.eps
        )
        return (y1 * y2).mean(dim=self.dim, **kwargs)


def odor_collate_fn(batch):
    dgl_graphs = [trial for b in batch for trial in b["odor_trials"]]
    inputs = dgl.batch(dgl_graphs).to(device)
    responses = torch.stack([trial for b in batch for trial in b["responses"]]).to(device)
    return inputs, responses


cfg = DEFAULT_CONFIG.copy()

with open_dict(cfg):
    cfg.dataset = {
        "cache_data": False,
    }
    cfg.dataset.modality_config = {"odor_trials": {}, "responses": {}}
    cfg.dataset.modality_config.odor_trials.interpolation = {
        "interpolation_mode": "nearest_neighbor"
    }
    cfg.dataset.modality_config.odor_trials.transforms = {}
    cfg.dataset.modality_config.odor_trials.transforms.ToSMILES = {
        "_target_": "includes.transforms.CIDToSMILESTransform"
    }
    cfg.dataset.modality_config.odor_trials.transforms.ToMol = {
        "_target_": "includes.transforms.SMILESToMolTransform"
    }
    cfg.dataset.modality_config.odor_trials.transforms.ToGraph = {
        "_target_": "includes.transforms.MolToGraphTransform"
    }
    cfg.dataset.modality_config.odor_trials.transforms.ToDGLGraph = {
        "_target_": "includes.transforms.GraphToDGLGraph",
        "self_loop": False,
    }
    cfg.dataset.modality_config.responses.interpolation = {
        "interpolation_window": 300,
        "interpolation_align": "left",
    }

    cfg.dataloader.num_workers = 0
    cfg.dataloader.prefetch_factor = None
    cfg.dataloader.batch_size = 128
    cfg.dataloader.pin_memory = False
    cfg.dataloader.shuffle = False
    cfg.dataloader.drop_last = False

print(OmegaConf.to_yaml(cfg))

train_dl = DataLoader(
    dataset=OdorDataset(full_paths[0], **cfg.dataset),
    collate_fn=odor_collate_fn,
    **cfg.dataloader,
)
sample = train_dl.dataset[0]
n_neurons = sample["responses"].shape[-1]

val_dl = train_dl
test_dl = train_dl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MPNNPOMWrapper(
    n_tasks=n_neurons,
    node_out_feats=100,
    edge_hidden_feats=75,
    edge_out_feats=100,
    num_step_message_passing=5,
    mpnn_residual=True,
    message_aggregator_type="sum",
    mode="regression",
    number_atom_features=GraphConvConstants.ATOM_FDIM,
    number_bond_features=GraphConvConstants.BOND_FDIM,
    readout_type="set2set",
    num_step_set2set=3,
    num_layer_set2set=2,
    ffn_hidden_list=[392, 392],
    ffn_embeddings=256,
    ffn_activation="relu",
    ffn_dropout_p=0.12,
    ffn_dropout_at_input_no_act=False,
).to(device)

seed = 1
nb_epoch = 200
epoch_log_frequency = 2
# learning_rate = ExponentialDecay(initial_rate=0.001, decay_rate=0.5, decay_steps=32*15, staircase=True)
learning_rate = 0.001
weight_decay = 1e-5
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
loss_fn = PoissonLoss()
corr_fn = PearsonCorrelation(dim=0)
use_wandb = True

filename = os.path.splitext(os.path.basename(__file__))[0]
wandb.init(
    project="odor-model",
    entity="rene-burghardt-foundation-models",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"{filename} ({os.environ['SLURM_JOB_ID']})" if "SLURM_JOB_ID" in os.environ else filename,
    # Track hyperparameters and run metadata
    config={
        "model": model.__class__.__name__,
        "dataset": {
            "dataset_path": paths,
            "dataloader": train_dl.__class__.__name__,
            "dataset": train_dl.dataset.__class__.__name__,
            "nr_train_batches": len(train_dl),
            "nr_val_batches": len(val_dl),
            "nr_test_batches": len(test_dl) if test_dl else 0,
            "batch_size": cfg.dataloader.batch_size,
        },
        "learning_rate": learning_rate,
        "max_epochs": nb_epoch,
        "custom": {},
    },
)

wandb.define_metric(name="Epoch", hidden=True)
wandb.define_metric(name="Batch", hidden=True)

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
start_time = datetime.now()
for epoch in range(1, nb_epoch + 1):
    model.train()
    avg_loss = 0.0
    averaged_batches = 0

    for inputs, responses in train_dl:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, responses)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        averaged_batches += 1

    train_loss = avg_loss / averaged_batches

    model.eval()
    avg_loss = 0.0
    averaged_batches = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, responses in val_dl:
            outputs = model(inputs)
            loss = loss_fn(outputs.detach(), responses)

            avg_loss += loss.item()
            averaged_batches += 1
            all_outputs.append(outputs)
            all_targets.append(responses)

    val_loss = avg_loss / averaged_batches

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    corr_value = corr_fn(all_outputs, all_targets).mean()

    if use_wandb:
        wandb_dict = {
            "Epoch": epoch,
            "Training loss": train_loss,
            "Validation correlation": corr_value,
            "Validation loss": val_loss,
        }
        wandb.log(wandb_dict)

    if epoch % epoch_log_frequency == 0 or epoch == 1 or epoch == nb_epoch:
        print(
            f"Epoch {epoch:03d}/{nb_epoch} ; Training loss: {train_loss:.4f} ; Validation loss: {val_loss:.4f} ; Validation correlation: {corr_value.item():.4f}"
        )

end_time = datetime.now()

print("time_taken: ", str(end_time - start_time))
model.eval()
avg_loss = 0.0
averaged_batches = 0
all_outputs = []
all_targets = []

with torch.no_grad():
    for inputs, responses in test_dl:
        outputs = model(inputs)
        loss = loss_fn(outputs.detach(), responses)

        avg_loss += loss.item()
        averaged_batches += 1
        all_outputs.append(outputs)
        all_targets.append(responses)

test_loss = avg_loss / averaged_batches

all_outputs = torch.cat(all_outputs, dim=0)
all_targets = torch.cat(all_targets, dim=0)

corr_value = corr_fn(all_outputs, all_targets).mean()
if use_wandb:
    wandb_dict = {
        "Test correlation": corr_value,
        "Test loss": test_loss,
    }
    wandb.log(wandb_dict)

print(f"Test correlation: {corr_value.item():.4f}")
