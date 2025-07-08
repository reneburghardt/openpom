import numpy as np
import pandas as pd
import pubchempy as pcp
from experanto.experiment import Experiment
from openpom.feat.graph_featurizer import GraphFeaturizer
from pyrfume import from_cids
from rdkit import Chem


def get_cid_from_smiles(smiles):
    try:
        compounds = pcp.get_compounds(smiles, namespace="smiles")
        return compounds[0].cid if compounds else np.nan
    except Exception as e:
        return np.nan


seed = 1

base_path = (
    "/user/burghardt17/u17926/.project/dir.project/rene/data/experanto-odor_trials/"
)
paths = [
    "030063__24_05_16",
    "030065__24_05_15",
    "030091__24_09_26",
    "030092__24_10_01",
    "030093__24_09_14",
    "030094__24_09_19",
]

modality_config = {
    "odor_trials": {"interpolation": {"interpolation_mode": "nearest_neighbor"}},
}

experiments = {path: Experiment(base_path + path, modality_config) for path in paths}
cids = []
for experiment in experiments.values():
    data = experiments["030063__24_05_16"].devices["odor_trials"]._data
    cids.append(np.unique(data[:, 0]))

cids = np.concatenate(cids)
cids = np.unique(cids)
cids = cids.astype(int)
cids = np.delete(cids, np.where(cids == 0))

np.set_printoptions(suppress=True)
print("CIDs in experimental data: ", cids)

info = from_cids(cids, property_list=["MolecularWeight", "SMILES", "IUPACName"])
molecules = pd.DataFrame(info)
smiles = molecules["SMILES"]

curated_dataset = pd.read_csv(
    "/user/burghardt17/u17926/.project/dir.project/rene/projects/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
)
curated_dataset["CID"] = curated_dataset["nonStereoSMILES"].apply(get_cid_from_smiles)
curated_SMILES = curated_dataset["nonStereoSMILES"]

overlapping_cids = molecules["CID"][molecules["CID"].isin(curated_dataset["CID"])]
print("Overlapping CIDs: ", overlapping_cids)

for cid in overlapping_cids:
    print("CID:", cid)
    print("IsomericSMILES:", molecules[molecules["CID"] == cid]["SMILES"].iloc[0])
    print(
        "nonStereoSMILES:",
        curated_dataset[curated_dataset["CID"] == cid]["nonStereoSMILES"].iloc[0],
    )

featurizer = GraphFeaturizer()
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
graphs = [featurizer._featurize(mol) for mol in mols]  # input format to openpom?!
