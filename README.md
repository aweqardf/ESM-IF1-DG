# ESM-IF1-DG

This repository contains a discriminator guided approach for functional protein inverse folding  

---


To set up a new conda environment with required packages,

```
conda create -n inverse python=3.9
conda activate inverse
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg -c conda-forge
conda install pip
pip install biotite
pip install git+https://github.com/facebookresearch/esm.git
```

# Quickstart
To sample sequences for a given structure in PDB or mmCIF format, use the
`sample_sequences.py` script. The input file can have either `.pdb` or
`.cif` as suffix.

For example, to sample 3 sequence designs for the golgi casein kinase structure
(PDB [5YH2](https://www.rcsb.org/structure/5yh2); [PDB Molecule of the Month
from January 2022](https://pdb101.rcsb.org/motm/265)), we can run the following
command from the `examples/inverse_folding` directory:

- The "--loss_type" should be chosen among ['Solubility', 'Stability', 'Both']


```
python sample_sequences.py data/5YH2.pdb \
    --chain C --temperature 1 --num-samples 3 \
    --outpath output/sampled_sequences.fasta \
    --loss_type Solubility \
    --stepsize 0.2 \
    --num_iterations 2 \
    --kl_scale 0.5 \
```
