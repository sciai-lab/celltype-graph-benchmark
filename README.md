# Plant CellTypeGraph Benchmark
A new graph benchmark for node classification

# dependencies
- linux
- python >= 3.7
- h5py
- pyaml
- pytorch
- pytorch_geometric
- torchmetrics

## Install dependencies using conda
- for cuda 11.1
```
conda create -n pctg -c rusty1s -c pytorch -c nvidia -c conda-forge numpy scipy matplotlib h5py pyaml tqdm pytorch torchvision cudatoolkit=11.1 pytorch-metrics pytorch-geometric
```
- for cuda 10.2
```
conda create -n pctg -c rusty1s -c pytorch -c conda-forge numpy scipy matplotlib h5py pyaml tqdm pytorch torchvision cudatoolkit=10.2 pytorch-metrics pytorch-geometric
```
- for cpu only 
```
conda create -n pctg -c rusty1s -c pytorch -c conda-forge numpy scipy matplotlib h5py pyaml tqdm pytorch torchvision cpuonly pytorch-metrics pytorch-geometric 
```
## Install Plant CellTypeGraph Benchmark
```
pip install .
```

## Examples
* create pytorch geometric dataset 
```
from pctg_benchmark.loaders import PCTGSimpleSplit
train_dataset = PCTGSimpleSplit(root='./', grs=('es_pca_grs', ), phase='train')
val_dataset = PCTGSimpleSplit(root='./', grs=('es_pca_grs', ), phase='val')
test_dataset = PCTGSimpleSplit(root='./', grs=('es_pca_grs', ), phase='test')
```
* Evaluate results
```
from pctg_benchmark.evaluation.metrics import NodeClassificationMetrics
eval_metrics = NodeClassificationMetrics(num_class=9)
eval_metrics.compute_metrics(predictions, target)
```