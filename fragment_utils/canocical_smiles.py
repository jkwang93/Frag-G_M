from rdkit import Chem
from rdkit.Chem import Draw

testsmi = 'ClC(C=C1)=CC=C1NC2=NC3=CC(OC4=C5C=C(C6=C(C)C=C(NC7CCN(CC7)C)C=C6)NC5=NC=N4)=CC=C3O2'
# testsmi = 'CCc1cc(-c2ccc(C)o2)n(-c2ccc3c(c2)nc(-c2cc(C(=O)N(C)C)ccc2O)n3Cc2ccc(OC)c(O)c2)n1'
mol = Chem.MolFromSmiles(testsmi)
for i in range(10):
    canonical_smi = Chem.MolToSmiles(mol)
    print(canonical_smi)
