# PaCPaC (Paratope and Clonotype Probing and Clustering)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4470165.svg)](https://doi.org/10.5281/zenodo.4470165)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Actions (Tests)](https://github.com/aretasg/pacpac/workflows/Tests/badge.svg)](https://github.com/aretasg/pacpac)

Python package to probe and cluster antibody paratopes and clonotypes

## :hammer_and_wrench: Installation and usage examples (Docker or Conda)
### :whale: Docker
#### Installation with Docker
You must have [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/) installed.
```bash
git clone https://github.com/aretasg/pacpac.git
cd pacpac
```
#### :computer: Example usage with Docker
Note the `csv_dataset` must be deposited in the `/data` folder.
```bash
docker-compose run pacpac cluster <csv_dataset> <vh_amino_acid_sequence_column_name>
docker-compose run pacpac probe <probe_vh_amino_acid_sequence> <csv_dataset> <vh_amino_acid_sequence_column_name>
```
Check `/data` folder for output
### :snake: Conda
#### Installation with Conda
Install [conda](https://docs.conda.io/en/latest/miniconda.html) first
```bash
git clone https://github.com/aretasg/pacpac.git
cd pacpac
conda env create -f environment.yml
conda activate pacpac
pip install .
```
#### :scroll: Example usage within Python
```python
import pandas as pd
from pacpac import pacpac

df = pd.read_csv(<my_data_set.csv>)

df = pacpac.cluster(df, <vh_amino_acid_sequence_column_name>)
df = pacpac.probe(<probe_vh_amino_acid_sequence>, df, <vh_amino_acid_sequence_column_name>)
```
or alternatively cluster and/or probe using both, VH and VL, sequences
```python
df = pacpac.cluster(df, <vh_amino_acid_sequence_column_name>, <vl_amino_acid_sequence_column_name>)
df = pacpac.probe(
  <probe_vh_amino_acid_sequence>,
  df,
  <vh_amino_acid_sequence_column_name>,
  <vl_amino_acid_sequence_column_name>,
  <probe_vl_amino_acid_sequence>
)
```
#### :computer: Example usage in CLI
```bash
pacpac cluster <path_to_csv_dataset> <vh_amino_acid_sequence_column_name>
pacpac probe <probe_vh_amino_acid_sequence> <path_to_csv_dataset> <vh_amino_acid_sequence_column_name>
```
## :question: Probing and clustering arguments
### within Python
```python
help(pacpac.cluster)
help(pacpac.probe)
```
### in CLI
```bash
pacpac cluster --help
pacpac probe --help
```

## :gem: Features
* Sequence annotations operations by [ANARCI](https://github.com/oxpig/ANARCI) (Dunbar and Deane, 2015).
* Deep learning model [Parapred](https://github.com/eliberis/parapred) for paratope predictions (Liberis et al., 2018).
* Clusters using greedy clustering approach.
* Determinism is achieved by sorting the input data set by CDR lengths and paratope length for clonotype and paratope clustering, respectively, and amino acid sequence in a descending order.
* Each cluster has a representitive sequence as indicated by a keyword `seed`.
* Clonotyping is done on the amino acid sequence level. Any silent mutations on nucleotide sequence level due to SHM are not taken into an account.
* Paratope probing and clustering provides several clustering options.

### Probing & Clustering options
* If `structural_equivalence` is set to `False` matches paratopes of equal CDR lengths only and assumes that CDRs of the same length always have deletions at the same position (Richardson et al., 2021). Useful in fast detection of similar paratopes.
* When set to `True` (default) structurally equivalence as assigned by the numbering scheme is used (i.e. numbering residue positions are used for residue matching to allow for a comparison at structuraly equivalent positions) and assumes that CDRs of different lengths can have similar paratopes. Useful in detection of similar binding modes.
* Sequence residues can be tokenized (`tokenize=True`) based on residue type groupings as described by Wong et al., 2021.

## :checkered_flag: Benchmarks with 10K VH sequences with 4 conventional CPU cores
| Task | Time (s) | Notes |
| -----------: | ----------------- | :----------: |
| Annotations using ANARCI | 378 | parallel execution |
| Paratope prediction using Parapred | 207 | parallel execution without CPU/GPU speed up for TensorFlow |
| Clonotype clustering | 13 | on amino acid level |
| Paratope clustering | 13 | `structural_equivalence=False` |
| Paratope clustering | 130 | `structural_equivalence=True` |
| Probing | <0.1 | clonotype & paratope |

ANARCI and Parapred can be speed up with more cores and/or CPU/GPU speed up instructions for Tensorflow.

## :pencil2: Authors
Written by **Aretas Gaspariunas**. Have a question? You can always ask and I can always ignore.

## References
- Dunbar and Deane, 2015
- Liberis et al., 2018
- Richardson et al., 2021
- Wong et al., 2021

## :apple: Citing
If you found PaCPaC useful for your work please acknowledge it by citing this repository.
```
@software{aretas_gaspariunas_2021_4470165,
  author       = {Aretas Gaspariunas},
  title        = {{aretasg/pacpac: PaCPaC - Python package to probe and cluster antibody VH sequence paratopes and clonotypes}},
  month        = jan,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.1},
  doi          = {10.5281/zenodo.4470165},
  url          = {https://doi.org/10.5281/zenodo.4470165}
}
```

## License
BSD license.
