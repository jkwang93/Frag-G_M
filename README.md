# Frag-G_M
Paper: Molecular Generation with Reduced Labeling through Constraint Architecture

## Environment
- python = 3.8.3
- pytroch = 1.6.0
- RDKit
- numpy
- pandas



## How to run？

```
# Train the proir conditional transformer
python train_prior.py

# generate condition molecules
python generator_Transformer.py --prior {piror_model_path} --save_molecules_path {save_molecules_path}

# generation the fragment vocabulary list
python ./fragment_utils/fragment.py

# train jnk+gsk+qed+sa agent model
python train_agent_save_smiles_jnk.py

```
