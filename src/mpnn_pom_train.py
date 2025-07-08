# %% [markdown]
# ### Example script for training MPNN-POM model

from datetime import datetime

import deepchem as dc

# %%
import torch
from openpom.feat.graph_featurizer import GraphConvConstants, GraphFeaturizer
from openpom.models.mpnn_pom import MPNNPOMModel
from openpom.utils.data_utils import get_class_imbalance_ratio

# %%
TASKS = [
    "alcoholic",
    "aldehydic",
    "alliaceous",
    "almond",
    "amber",
    "animal",
    "anisic",
    "apple",
    "apricot",
    "aromatic",
    "balsamic",
    "banana",
    "beefy",
    "bergamot",
    "berry",
    "bitter",
    "black currant",
    "brandy",
    "burnt",
    "buttery",
    "cabbage",
    "camphoreous",
    "caramellic",
    "cedar",
    "celery",
    "chamomile",
    "cheesy",
    "cherry",
    "chocolate",
    "cinnamon",
    "citrus",
    "clean",
    "clove",
    "cocoa",
    "coconut",
    "coffee",
    "cognac",
    "cooked",
    "cooling",
    "cortex",
    "coumarinic",
    "creamy",
    "cucumber",
    "dairy",
    "dry",
    "earthy",
    "ethereal",
    "fatty",
    "fermented",
    "fishy",
    "floral",
    "fresh",
    "fruit skin",
    "fruity",
    "garlic",
    "gassy",
    "geranium",
    "grape",
    "grapefruit",
    "grassy",
    "green",
    "hawthorn",
    "hay",
    "hazelnut",
    "herbal",
    "honey",
    "hyacinth",
    "jasmin",
    "juicy",
    "ketonic",
    "lactonic",
    "lavender",
    "leafy",
    "leathery",
    "lemon",
    "lily",
    "malty",
    "meaty",
    "medicinal",
    "melon",
    "metallic",
    "milky",
    "mint",
    "muguet",
    "mushroom",
    "musk",
    "musty",
    "natural",
    "nutty",
    "odorless",
    "oily",
    "onion",
    "orange",
    "orangeflower",
    "orris",
    "ozone",
    "peach",
    "pear",
    "phenolic",
    "pine",
    "pineapple",
    "plum",
    "popcorn",
    "potato",
    "powdery",
    "pungent",
    "radish",
    "raspberry",
    "ripe",
    "roasted",
    "rose",
    "rummy",
    "sandalwood",
    "savory",
    "sharp",
    "smoky",
    "soapy",
    "solvent",
    "sour",
    "spicy",
    "strawberry",
    "sulfurous",
    "sweaty",
    "sweet",
    "tea",
    "terpenic",
    "tobacco",
    "tomato",
    "tropical",
    "vanilla",
    "vegetable",
    "vetiver",
    "violet",
    "warm",
    "waxy",
    "weedy",
    "winey",
    "woody",
]
print("No of tasks: ", len(TASKS))

# The curated dataset can also found at `openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv` in the repo.

input_file = "/user/burghardt17/u17926/.project/dir.project/rene/projects/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"  # or new downloaded file path

# %%
# get dataset

featurizer = GraphFeaturizer()
smiles_field = "nonStereoSMILES"
loader = dc.data.CSVLoader(
    tasks=TASKS, feature_field=smiles_field, featurizer=featurizer
)
dataset = loader.create_dataset(inputs=[input_file])
n_tasks = len(dataset.tasks)

# %%
len(dataset)

# %%
# get train valid test splits

randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
train_dataset, test_dataset, valid_dataset = (
    randomstratifiedsplitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=1
    )
)

# %%
print("train_dataset: ", len(train_dataset))
print("valid_dataset: ", len(valid_dataset))
print("test_dataset: ", len(test_dataset))


# %%
train_ratios = get_class_imbalance_ratio(train_dataset)
assert len(train_ratios) == n_tasks

# %%
# learning_rate = ExponentialDecay(initial_rate=0.001, decay_rate=0.5, decay_steps=32*15, staircase=True)
learning_rate = 0.001

# %%
# initialize model

model = MPNNPOMModel(
    n_tasks=n_tasks,
    batch_size=128,
    learning_rate=learning_rate,
    class_imbalance_ratio=train_ratios,
    loss_aggr_type="sum",
    node_out_feats=100,
    edge_hidden_feats=75,
    edge_out_feats=100,
    num_step_message_passing=5,
    mpnn_residual=True,
    message_aggregator_type="sum",
    mode="classification",
    number_atom_features=GraphConvConstants.ATOM_FDIM,
    number_bond_features=GraphConvConstants.BOND_FDIM,
    n_classes=1,
    readout_type="set2set",
    num_step_set2set=3,
    num_layer_set2set=2,
    ffn_hidden_list=[392, 392],
    ffn_embeddings=256,
    ffn_activation="relu",
    ffn_dropout_p=0.12,
    ffn_dropout_at_input_no_act=False,
    weight_decay=1e-5,
    self_loop=False,
    optimizer_name="adam",
    log_frequency=32,
    model_dir="./examples/experiments",
    device_name="cuda",
)

# %%
nb_epoch = 200

# %%
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)


# %%
start_time = datetime.now()
for epoch in range(1, nb_epoch + 1):
    breakpoint()
    loss = model.fit(
        train_dataset,
        nb_epoch=1,
        max_checkpoints_to_keep=1,
        deterministic=False,
        restore=epoch > 1,
    )
    train_scores = model.evaluate(train_dataset, [metric])["roc_auc_score"]
    valid_scores = model.evaluate(valid_dataset, [metric])["roc_auc_score"]
    print(
        f"epoch {epoch}/{nb_epoch} ; loss = {loss}; train_scores = {train_scores}; valid_scores = {valid_scores}"
    )
model.save_checkpoint()
end_time = datetime.now()

# %%
test_scores = model.evaluate(test_dataset, [metric])["roc_auc_score"]
print("time_taken: ", str(end_time - start_time))
print("test_score: ", test_scores)
