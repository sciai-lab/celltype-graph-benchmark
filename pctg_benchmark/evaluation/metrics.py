from torchmetrics import Accuracy, Precision, Recall, F1
from dataclasses import dataclass


@dataclass
class NodeClassificationMetrics:
    num_classes: int

    def __post_init__(self):
        self.metrics = {'accuracy_micro': Accuracy(average='micro'),
                        'accuracy_macro': Accuracy(num_classes=self.num_classes, average='macro'),
                        'accuracy_class': Accuracy(num_classes=self.num_classes, average=None),
                        'precision_micro': Precision(average='micro'),
                        'precision_class': Precision(num_classes=self.num_classes, average=None),
                        'recall_micro': Recall(average='micro'),
                        'recall_class': Recall(num_classes=self.num_classes, average=None),
                        'f1_micro': F1(average='micro'),
                        'f1_class': F1(num_classes=self.num_classes, average=None)}

    def compute_metrics(self, pred, target, step=0):
        results = {}
        for key, metric in self.metrics.items():
            value = metric(pred, target)
            results[key] = value
            if value.ndim == 0:
                value = value.item()

            elif value.ndim == 1:
                value = [v.item() for v in value]

            elif value.ndim == 2:
                value = [[v1.item() for v1 in v2] for v2 in value]

            else:
                raise NotImplementedError

            results[key] = value
        results['step'] = step
        return results
