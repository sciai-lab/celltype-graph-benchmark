# CellTypeGraph Benchmark
A new graph benchmark for node classification

## Requirements
- Linux
- Anaconda / miniconda

### Dependencies
- python >= 3.8
- numpy
- tqdm
- h5py
- requests
- pyyaml
- numba
- pytorch
- torchmetrics
- pytorch-geometric

### Optional Dependencies (for running the examples):
- class_resolver

## Install CellTypeGraph Benchmark using conda
- for cuda 11.3
```
conda create -n ctg -c rusty1s -c pytorch -c conda-forge -c lcerrone ctg-benchmark cudatoolkit=11.3
```
- for cuda 10.2
```
conda create -n ctg -c rusty1s -c pytorch -c conda-forge -c lcerrone ctg-benchmark cudatoolkit=10.2
```
- for cpu only
```
conda create -n ctg -c rusty1s -c pytorch -c conda-forge -c lcerrone ctg-benchmark cpuonly 
```
# Simple training example
* A simple GCN training example can be found in [examples](examples/gcn_example.py).

## Basic usage
* create CellTypeGraph cross validation loader
```python
from ctg_benchmark.loaders.torch_loader import get_cross_validation_loaders
loaders_dict = get_cross_validation_loaders(root='./ctg_data/')
```
where `loaders_dict` is a dictionary that contains 5 tuple of training and validation data-loaders. 
```python
for split, loader_dict in loaders_dict.items():
    train_loader = loader_dict['train'] 
    val_loader = loader_dict['val']
```


* Alternatively for quicker experimentation's one can create a simples train/val/test split as: 
```python
from ctg_benchmark.loaders.torch_loader import get_split_loaders
loader = get_split_loaders(root='./ctg_data/',)
print(loader['train'], loader['val'], loader['test'])
```

* Simple evaluation: For evaluation `ctg_benchmark.evaluation.NodeClassificationMetrics` conveniently wraps several 
metrics as implemented in`torchmetrics`. Single class results can be 
aggregate by using `ctg_benchmark.evaluation.aggregate_class`.
```python
from ctg_benchmark.evaluation.metrics import NodeClassificationMetrics, aggregate_class
eval_metrics = NodeClassificationMetrics(num_classes=9)

predictions = torch.randint(9, (1000,))
target = torch.randint(9, (1000,))
results = eval_metrics.compute_metrics(predictions, target)
class_average_accuracy, _ = aggregate_class(results['accuracy_class'], index=7)

print(f"global accuracy: {results['accuracy_micro']: .3f}")
print(f"class average accuracy: {class_average_accuracy: .3f}")
```

## Advanced usage examples
* Change default features and features processing
* Load points samples
* Add features
* Manual download

## Reproducibility 
* To get details on the benchmark, run the following [script](examples/benchmark_overview.py)
* To reproduce the GCN results checkout the following [script](examples/gcn_reproducibility.py)
* To reproduce all baseline results, plots and additional experiments, checkout the 
[plantcelltype](https://github.com/hci-unihd/plant-celltype) repository.

## Cite
coming soon...