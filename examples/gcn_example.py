from ctg_benchmark.loaders.torch_loader import get_cross_validation_loaders
from ctg_benchmark.evaluation.metrics import NodeClassificationMetrics, aggregate_class
import torch
import numpy as np
from tqdm import trange
from torch_geometric.nn.models import GCN
from torch.optim import Adam
import torch.nn.functional as F


def simple_trainer(trainer_loader):
    model = GCN(in_channels=74, hidden_channels=128, num_layers=2, out_channels=9, dropout=0.5)
    optim = Adam(params=model.parameters(), lr=1e-2, weight_decay=1e-5)
    t_range = trange(100, desc=f'Epoch: {0: 03d}, training loss: {0/len(trainer_loader): .2f}')
    for epoch in t_range:
        loss_epoch = 0

        for batch in trainer_loader:
            optim.zero_grad()

            pred = model.forward(batch.x, batch.edge_index)
            logits = torch.log_softmax(pred, 1)
            loss = F.nll_loss(logits, batch.y)
            loss.backward()

            optim.step()

            loss_epoch += loss.item()

        t_range.set_description(f'Epoch: {epoch + 1: 03d}, training loss: {loss_epoch/len(trainer_loader): .2f}')
        t_range.refresh()
    return model


def validation(validation_loader, model):
    # set up evaluation
    eval_metrics = NodeClassificationMetrics(num_classes=9)

    accuracy_records, accuracy_class_records = [], []
    model.eval()
    with torch.no_grad():
        for val_batch in validation_loader:
            pred = model.forward(val_batch.x, val_batch.edge_index)
            logits = torch.log_softmax(pred, 1)
            pred = logits.max(1)[1]

            # results is a dictionary containing a large number of classification metrics
            results = eval_metrics.compute_metrics(pred, val_batch['y'])
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

        model = simple_trainer(training_loader)
        split_accuracy_records, split_accuracy_class_records = validation(validation_loader, model)
        accuracy_records += split_accuracy_records
        accuracy_class_records += split_accuracy_class_records

    # report results
    print(f'\nGCN results:')
    print(f'Accuracy {np.mean(accuracy_records):.3f} std: {np.std(accuracy_records):.3f}')
    print(f'Class Accuracy {np.mean(accuracy_class_records):.3f} std: {np.std(accuracy_class_records):.3f}')


if __name__ == '__main__':
    main()
