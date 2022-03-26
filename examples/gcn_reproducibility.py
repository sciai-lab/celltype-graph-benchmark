"""
In order to reproduce the baselines scores one need to:
* Downloads `baseline_checkpoints.zip` from https://heibox.uni-heidelberg.de/published/celltypegraph-benchmark/
and extract `TgGCN_lr_1e-2_wd_1e-5_num_layers_2_hidden_feat_128_dropout_0.5` here.
(zip file check `md5sum: 82ea2e1cf3d0e9eee342938b7c583174`).
"""

from ctg_benchmark.loaders.torch_loader import get_cross_validation_loaders
from ctg_benchmark.evaluation.metrics import NodeClassificationMetrics, aggregate_class
import torch
import numpy as np
from torch_geometric.nn.models import GCN
import glob
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = f'./TgGCN_lr_1e-2_wd_1e-5_num_layers_2_hidden_feat_128_dropout_0.5/'


def load_gcn(split=0):
    model = GCN(in_channels=74, hidden_channels=128, num_layers=2, out_channels=9, dropout=0.5)

    path = f'{base_path}/split{split}/version_0/checkpoints/best_acc*.ckpt'
    path = glob.glob(path)[0]
    _model = torch.load(path)
    state_dict = _model['state_dict']
    new_state_dict = OrderedDict([(key.replace('net.module.', ''), value) for key, value in state_dict.items()])
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    return model


def validation(validation_loader, model):
    # set up evaluation
    eval_metrics = NodeClassificationMetrics(num_classes=9)

    accuracy_records, accuracy_class_records = [], []
    model.eval()
    with torch.no_grad():
        for val_batch in validation_loader:
            val_batch = val_batch.to(device)
            pred = model.forward(val_batch.x, val_batch.edge_index)
            logits = torch.log_softmax(pred, 1)
            pred = logits.max(1)[1]

            # results is a dictionary containing a large number of classification metrics
            results = eval_metrics.compute_metrics(pred.cpu(), val_batch.y.cpu())
            acc = results['accuracy_micro']
            # aggregate class average the single class accuracy and ignores the embryo sack class (7)
            acc_class, _ = aggregate_class(results['accuracy_class'], index=7)

            accuracy_records.append(acc)
            accuracy_class_records.append(acc_class)
    return accuracy_records, accuracy_class_records


def main():
    # create data loader
    loader = get_cross_validation_loaders(root='./ctg_data', batch_size=1, shuffle=True, grs=('label_grs_surface',))

    accuracy_records, accuracy_class_records = [], []
    for split, split_loader in loader.items():
        training_loader, validation_loader = split_loader['train'], split_loader['val']

        model = load_gcn(split)
        split_accuracy_records, split_accuracy_class_records = validation(validation_loader, model)
        accuracy_records += split_accuracy_records
        accuracy_class_records += split_accuracy_class_records

    # report results
    print(f'\nGCN results:')
    print(f'Accuracy {np.mean(accuracy_records):.3f} std: {np.std(accuracy_records):.3f}')
    print(f'Class Accuracy {np.mean(accuracy_class_records):.3f} std: {np.std(accuracy_class_records):.3f}')


if __name__ == '__main__':
    main()
