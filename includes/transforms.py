from pyrfume import from_cids
import numpy as np
from rdkit import Chem
from openpom.feat.graph_featurizer import GraphFeaturizer


class CIDToSMILESTransform():
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
        self.property_name = "SMILES"

    def __call__(self, odor_concentration_matrix):
        results = from_cids(odor_concentration_matrix[:, self.dim].astype(int), property_list=[self.property_name])
        return [d[self.property_name] for d in results]


class SMILESToMolTransform():
    def __call__(self, smiles):
        return [Chem.MolFromSmiles(s) for s in smiles]


class MolToGraphTransform():
    def __init__(self):
        super().__init__()
        self.featurizer = GraphFeaturizer()

    def __call__(self, mols):
        return [self.featurizer(m)[0] for m in mols]

class GraphToDGLGraph():
    def __init__(self, self_loop=False):
        super().__init__()
        self.self_loop = self_loop

    def __call__(self, graphs):
        return [graph.to_dgl_graph(self_loop=self.self_loop) for graph in graphs]