# Source code for the implementation of the paper: "Leveraging patient similarities via graph neural networks to predict phenotypes from temporal data" https://ieeexplore.ieee.org/document/10302556

# Replicate benchmark 
1. This replication of this source code requires [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) dataset. You will need to pass the following examination [CITI](https://physionet.org/about/citi-course/) to access the [Physionet ](https://physionet.org/) data repository: 
2. For replication of the benchmark you can follow the instructions at:
- https://github.com/YerevaNN/mimic3-benchmarks


## Benchmark replication 
<pre><code>
conda create -n "mimic3" python=3.7.13
conda activate mimic3
pip install -r requirements.txt
wget -r -N -c -np https://physionet.org/files/mimiciii-demo/1.4/ # this will create a physionet folder with the Database csvs
python -m mimic3benchmark.scripts.extract_subjects physionet.org/files/mimiciii-demo/1.4/ data/root
python -m mimic3benchmark.scripts.validate_events data/root/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
python -m mimic3benchmark.scripts.split_train_and_test data/root/



python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/

python -m mimic3models.split_train_val data/phenotyping

python -um mimic3models.phenotyping.logistic.main --output_dir mimic3models/phenotyping/logistic
# ls data/phenotyping/statistical_features 
# test_X          test_ts         train_X         train_ts        val_X           val_ts
# test_names      test_y          train_names     train_y         val_names       val_y

# The libraries used for these LSTM are highly problematic and require specific version -> torch based equivalent of SCEHR has been used insted
# python -um mimic3models.phenotyping.main --network mimic3models/keras_models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/phenotyping
python -m mimic3benchmark.evaluation.evaluate_phenotyping data/phenotyping/train data/phenotyping/train predictions/phenotyping/logistic train


# We used an alternative pytorch training of LSTM training and eval using SCEHR implementation based on SCEHR
python -m mimic3models.train_lstm --network mimic3models/lstm.py --data data/phenotyping/ --save

</code>
</pre>

# GNN training 

## environment setup
The libraries and python version are more recent than those used for the benchmark. Please raise an issue if you find difficulty in any of the steps below. 


## create graphs
<pre><code>
conda env create -f environment.yaml
python create_homogeneous_graphs.py --edge_strategy trivial --node_embeddings_type stat --folder_name graphs
python create_homogeneous_graphs.py --edge_strategy random --node_embeddings_type stat --folder_name graphs
python create_homogeneous_graphs.py --edge_strategy expert_exact --node_embeddings_type stat --folder_name graphs
python create_homogeneous_graphs.py --edge_strategy knn_graph --node_embeddings_type stat --folder_name graphs
python create_homogeneous_graphs.py --edge_strategy expert_medium --node_embeddings_type stat --folder_name graphs
python create_homogeneous_graphs.py --edge_strategy expert_lenient --node_embeddings_type stat --folder_name graphs
'random', 'expert_exact', 'expert_medium', 'expert_lenient', 'knn_graph', 'trivial' 
</code></pre>
## expert rules connectivity strategies 
<pre><code>
python -m gnn__models.connectivity_strategies.expert_graph_m1_exact
python -m gnn__models.connectivity_strategies.expert_graph_m2_inter_category
python -m gnn__models.connectivity_strategies.expert_graph_m3_intracategory
</code></pre>


## Example gnn training 
<pre><code>
python train_gnn.py --model SAGEConv --data_folder graphs/data_trivial_stat/processed/ --epochs 1 --WD 0.001 --lr 0.0001 --hidden 8192 --batch_size 512 --model_name SAGEConv_nf_stat_es_knn_2011_05_19_13_55_26 --mode_training transductive --model_folder graph_model --experiment_name exp_v1_trivial_stat 
</code></pre>
