from torchmetrics import Accuracy, Precision, Recall, F1
from dataclasses import dataclass
from typing import Optional


@dataclass
class NodeClassificationMetrics:
    num_classes: int
    ignore_index: Optional[int] = None

    def __post_init__(self):
        self.metrics = {'accuracy_micro': Accuracy(average='micro',
                                                   ignore_index=self.ignore_index),
                        'accuracy_macro': Accuracy(num_classes=self.num_classes,
                                                   average='macro',
                                                   ignore_index=self.ignore_index),
                        'accuracy_class': Accuracy(num_classes=self.num_classes,
                                                   average=None,
                                                   ignore_index=self.ignore_index),
                        'precision_micro': Precision(average='micro', ignore_index=self.ignore_index),
                        'precision_class': Precision(num_classes=self.num_classes,
                                                     average=None,
                                                     ignore_index=self.ignore_index),
                        'recall_micro': Recall(average='micro', ignore_index=self.ignore_index),
                        'recall_class': Recall(num_classes=self.num_classes,
                                               average=None,
                                               ignore_index=self.ignore_index),
                        'f1_micro': F1(average='micro', ignore_index=self.ignore_index),
                        'f1_class': F1(num_classes=self.num_classes,
                                       average=None,
                                       ignore_index=self.ignore_index)
                        }

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
