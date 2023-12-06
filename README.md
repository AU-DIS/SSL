# SSL
This is a repository for the <i>**Spectral Subgraph Localization**</i> paper. The algorithm is implemented in Python3.


**Packages Installation**
```sh
cd ./SSL\
pipenv install
```

Install the dependencies in requirements.txt
```sh
pip3 install -r requirements.txt
```

**Graph Generator**
```sh
python3 create_dataset.py
```
**Query Subgraphs Genarator**
```sh
python3 graph_generator.py
```

# Experiments
The experiments can be reproduced by running the following commands:
**SSL Experiments - All Methods**
```sh
python3 experiments_ssl_all_methods.py dataset percentage_lower_bound percentage_upper_bound per edge_removal folder_number
```
Example : python3 experiments_ssl_all_methods.py football 1 11 0.1 0.3 1

dataset is the dataset you would like, eg. 'football'.

percentage_lower_bound is the lower bound of which of the .txt files you would like to target.

percentage_upper_bound similarly is the upper bound of the .txt files.

per is either 0.1, 0.2 or 0.3. This targets the ratio of query graph size V_Q/V folders immediately after the initial dataset folder.

edge_removal is the percentage of edges you remove in the initial edge-removing stage of our algorithm.

folder_number is the specific folder of the dataset you want to run the experiment on

**Final solution - Voting & Neighbourhood**
```sh
python3 get_increasing_edge_removal.py football 0.1 1 
```

**SSL Experiments - Data Term**
```sh
python3 experiments_ssl_data_term.py football 1 11 0.1 0.3 1
```

**CONE Experiments**
```sh
python3 CONE/experiments_CONE.py
```

**Spectrum Experiments**
```sh
python3 experiments_spectrum.py
```

**Convert PKL to CSV format**
```sh
python3 pkl_to_csv.py
```

**SSL data to CSV format**
```sh
python3 plot_average_spectrum_diff.py football 0.1 0.2 0.2 3 balanced_accuracy
```

**CONE data to CSV format**
```sh
python3 plot_CONE.py football 0.1
```

## Citation
```sh
@inproceedings{
bainson2023spectral,
title={Spectral Subgraph Localization},
author={Ama Bembua Bainson and Judith Hermanns and Petros Petsinis and Niklas Aavad and Casper Dam Larsen and Tiarnan Swayne and Amit Boyarski and Davide Mottin and Alex M. Bronstein and Panagiotis Karras},
booktitle={The Second Learning on Graphs Conference},
year={2023},
url={https://openreview.net/forum?id=zrOMpghV0M}
}
```