# SSL
This is a repository for the <i>**Spectral Subgraph Localization**</i> paper.

**Packages Installation**\
cd ./SSL\
pipenv install

**Graph Generator**\
python3 create_dataset.py

**Query Subgraphs Genarator**\
python3 graph_generator.py


**SSL Experiments - All Methods**\
python3 experiments_ssl_all_methods.py dataset percentage_lower_bound percentage_upper_bound per edge_removal folder_number
Example : python3 experiments_ssl_all_methods.py football 1 11 0.1 0.3 1

dataset is the dataset you would like, eg. 'football'.

percentage_lower_bound is the lower bound of which of the .txt files you would like to target.

percentage_upper_bound similarly is the upper bound of the .txt files.

per is either 0.1, 0.2 or 0.3. This targets the ratio of query graph size V_Q/V folders immediately after the initial dataset folder.

edge_removal is the percentage of edges you remove in the initial edge-removing stage of our algorithm.

folder_number is the specific folder of the dataset you want to run the experiment on

**Final solution - Voting & Neighbourhood**\
python3 get_increasing_edge_removal.py football 0.1 1 

**SSL Experiments - Data Term**\
python3 experiments_ssl_data_term.py football 1 11 0.1 0.3 1

**CONE Experiments**\
python3 .CONE/experiments_CONE.py

**Spectrum Experiments**\
python3 experiments_spectrum.py

**Convert PKL to CSV format**\
python3 pkl_to_csv.py

**CONE data to CSV format**\
python3 plot_CONE.py football 0.1
