from ctg_benchmark.loaders.torch_loader import get_cross_validation_loaders
from ctg_benchmark.evaluation.metrics import NodeClassificationMetrics, aggregate_class
import torch
import numpy as np


def mock_predictor(val_batch):
    from ctg_benchmark.utils.io import open_full_stack
    from ctg_benchmark import gt_mapping
    """ Simple mock predictions module, load the experts annotations and uses it as predictions """
    stack = open_full_stack(val_batch['file_path'][0], keys=('cell_labels_set2',))

    pred = [gt_mapping.get(i, 0) for i in stack['cell_labels_set2']]
    # deal with few exceptions (missing entries in the new annotations are mapped to es)
    pred = [i if i is not None else 7 for i in pred]
    pred = torch.Tensor(pred).long()
    return pred


def mock_trainer(*args):
    # TODO to be implemented
    ...


def main():
    # create data loader
    loader = get_cross_validation_loaders(root='./ctg_data', batch_size=1, shuffle=True, grs=('label_grs_surface',))

    # set up evaluation
    eval_metrics = NodeClassificationMetrics(num_classes=9)

    accuracy_records, accuracy_class_records = [], []
    num_edges, num_nodes = 0, 0

    print("Running template cross-validation loop:")
    for split, split_loader in loader.items():
        training_loader, validation_loader = split_loader['train'], split_loader['val']
        # TODO your training procedure goes here
        mock_trainer(training_loader, validation_loader)

        print(f' Cross validation split: {split + 1}/{len(loader)}')
        for val_batch in validation_loader:
            # TODO your predictor goes here
            pred = mock_predictor(val_batch)

            # results is a dictionary containing a large number of classification metrics
            results = eval_metrics.compute_metrics(pred, val_batch['y'])
            acc = results['accuracy_micro']
            # aggregate class average the single class accuracy and ignores the embryo sack class (7)
            acc_class, _ = aggregate_class(results['accuracy_class'], index=7)

            accuracy_records.append(acc)
            accuracy_class_records.append(acc_class)

            # collect benchmark statistics
            num_edges += val_batch.num_edges
            num_nodes += val_batch.num_nodes

    # report results
    print('\nDataset statistics:')
    print(f' #specimen {len(accuracy_records)}')
    print(f' #features {val_batch.x.shape[-1]}, #edge features {val_batch.edge_attr.shape[-1]}')
    print(f' #edges: {num_edges}, #nodes: {num_nodes}')
    print(f' <#edges>: {num_edges // 84}, <#nodes>: {num_nodes // 84}')

    print(f'\nExpert performance:')
    print(f' Accuracy {np.mean(accuracy_records):.3f} {np.std(accuracy_records):.3f}')
    print(f' Class Accuracy {np.mean(accuracy_class_records):.3f} {np.std(accuracy_class_records):.3f}')


if __name__ == '__main__':
    main()
