# CellTypeGraph Benchmark
A new graph benchmark for node classification

# Requirements
- Linux
- Anaconda / miniconda

# dependencies
- python >= 3.7
- h5py
- pyaml
- pytorch
- pytorch_geometric
- torchmetrics

## Install dependencies using conda
- for cuda 11.1
```
conda create -n ctg -c rusty1s -c pytorch -c nvidia -c conda-forge numpy scipy matplotlib h5py pyaml tqdm pytorch torchvision cudatoolkit=11.1 pytorch-metrics pytorch-geometric
```
- for cuda 10.2
```
conda create -n ctg -c rusty1s -c pytorch -c conda-forge numpy scipy matplotlib h5py pyaml tqdm pytorch torchvision cudatoolkit=10.2 pytorch-metrics pytorch-geometric
```
- for cpu only
```
conda create -n ctg -c rusty1s -c pytorch -c conda-forge numpy scipy matplotlib h5py pyaml tqdm pytorch torchvision cpuonly pytorch-metrics pytorch-geometric 
```
## Install CellTypeGraph Benchmark
From inside the project root:
```
pip install .
```

## Minimal Example
* create CellTypeGraph simple split loader: 
```
from ctg_benchmark.loaders.torch_loader import get_split_loaders
loader = get_split_loaders(root='./ctg_data/',)
print(loader['train'], loader['val'], loader['test'])
```

* create CellTypeGraph cross validation loader:
```
from ctg_benchmark.loaders.torch_loader import get_cross_validation_loaders
loader = get_cross_validation_loaders(root='./ctg_data/',)
for split, loader in loader.items():
    print(split, loader['train'], loader['val'])
```

* Evaluate results:
```
from ctg_benchmark.evaluation.metrics import NodeClassificationMetrics
eval_metrics = NodeClassificationMetrics(num_classes=9)

predictions = torch.randint(9, (1000,))
target = torch.randint(9, (1000,))
results = eval_metrics.compute_metrics(predictions, target)
class_average_accuracy, _ = aggregate_class(results['accuracy_class'], index=7)

print(f"global accuracy: {results['accuracy_micro']: .3f}")
print(f"class average accuracy: {class_average_accuracy: .3f}")
```
