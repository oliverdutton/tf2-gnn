from typing import Any, Dict, Iterable, List, NamedTuple, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf

from tf2_gnn.data import GraphDataset
from tf2_gnn.models import NodeMulticlassTask
from tf2_gnn.utils.metrics import micro_f1


class SudokuTask(NodeMulticlassTask):
    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None):
        super().__init__(params, dataset, name)

    def compute_task_metrics(
        self, batch_features, task_output, batch_labels
    ) -> Dict[str, tf.Tensor]:
        per_node_logits = tf.convert_to_tensor(task_output)[0]
        (loss, f1_score) = self._fast_task_metrics(per_node_logits[-1], batch_labels["node_labels"])
        task_metrics = {"loss": loss, "f1_score": f1_score}

        sudoku_metrics = self._compute_sudoku_metrics(per_node_logits, batch_labels["node_labels"])
        task_metrics.update(sudoku_metrics)

        if self._loss_at_every_layer:
            """ If _loss_at_every_layer the f1_score for the final layer is returned but loss is calculated 
                    as the mean loss of each layer"""          
            every_layer_loss = tf.reduce_mean([self._fast_task_metrics(per_node_logits_per_layer, batch_labels["node_labels"])[0] for per_node_logits_per_layer in per_node_logits])
            task_metrics.update({"loss":every_layer_loss, "final_layer_loss":loss})
        return task_metrics

    @staticmethod
    def _compute_sudoku_metrics(per_node_logits, node_labels) -> Dict[str, tf.Tensor]:
        n_nodes = per_node_logits.shape[1]
        n_sudokus = int(per_node_logits.shape[1] / 81)

        answers_one_hot = tf.reshape(node_labels, (n_sudokus,81,9))
        answers_values = tf.argmax(input=answers_one_hot, axis=-1, output_type=tf.int32)+1

        node_predictions_logits = tf.convert_to_tensor(per_node_logits)
        n_layers = node_predictions_logits.shape[0]
        sudoku_predictions_logits = tf.reshape(node_predictions_logits, (n_layers,n_sudokus,81,9))
        sudoku_predictions_values = tf.argmax(input=sudoku_predictions_logits, axis=-1, output_type=tf.int32)+1

        digits_correct = tf.equal(sudoku_predictions_values, answers_values)
        digit_accs = tf.math.count_nonzero(digits_correct, axis=[1,2], dtype=tf.float32) / n_nodes

        sudokus_correct = tf.reduce_all(digits_correct, axis=-1)
        sudoku_accs = tf.math.count_nonzero(sudokus_correct, axis=1, dtype=tf.float32) / n_sudokus

        return {'digit_accuracies_per_layer':digit_accs, 'final_layer_digit_accuracy':digit_accs[-1], 
        'sudoku_accuracies_per_layer':sudoku_accs, 'final_layer_sudoku_accuracy':sudoku_accs[-1]}

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        avg_microf1 = np.average([r["f1_score"] for r in task_results])
        avg_digit_acc = np.average([r["final_layer_digit_accuracy"] for r in task_results])
        avg_sudoku_acc = np.average([r["final_layer_sudoku_accuracy"] for r in task_results])
        return -avg_digit_acc, f"AvgDigitAccuracy: {avg_digit_acc:.4f}, AvgSudokuAccuracy: {avg_sudoku_acc:.4f},\n Avg MicroF1: {avg_microf1:.4f}"
