Source code for the implementation of the paper: 

# replicate benchmark 
"Leveraging patient similarities via graph neural networks to predict phenotypes from temporal data"

1. This replication of this source code requires [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) dataset. You will need to pass the following examination [CITI](https://physionet.org/about/citi-course/) to access the [Physionet ](https://physionet.org/) data repository: 
2. For replication of the benchmark you can follow the instructions at:
- https://github.com/YerevaNN/mimic3-benchmarks


## TLDR benchmark replication 
<pre><code>
conda create -n "mimic3" python=3.7.13
conda activate mimic3
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

# The libraries used for these LSTM are highly problematic and require specific version -> torch based equivalent of SCEHR has been used insted
# python -um mimic3models.phenotyping.main --network mimic3models/keras_models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/phenotyping

python -m mimic3benchmark.evaluation.evaluate_phenotyping data/phenotyping/train data/phenotyping/train predictions/phenotyping/logistic train

</code>
</pre>

# GNN training 

## environment setup
The libraries and python version are more recent than those used for the benchmark. Please raise an issue if you find difficulty in any of the steps below. 
 

<pre><code>
conda env create -f environment.
conda activate cloned_env
</code></pre>

## expert rules connectivity strategies 
<pre><code>

python -m gnn__models.connectivity_strategies.expert_graph_m1_exact
python -m gnn__models.connectivity_strategies.expert_graph_m2_inter_category
python -m gnn__models.connectivity_strategies.expert_graph_m3_intracategory
</code></pre>

## Node features 
### Statistical moments 
Dimensionality: 714 = 6 subperiods x 7 moments x 17 vital signs 
<pre><code>

</code></pre>


### LSTM hidden space 
Dimentionality: 256 from bidirectional LSTM 