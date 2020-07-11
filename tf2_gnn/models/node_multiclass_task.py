from typing import Any, Dict, Iterable, List, NamedTuple, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf

from tf2_gnn.data import GraphDataset
from tf2_gnn.models import GraphTaskModel


def micro_f1(logits, labels):
    # Everything on int, because who trusts float anyway?
    predicted = tf.math.round(tf.nn.sigmoid(logits))
    predicted = tf.cast(predicted, dtype=tf.int32)
    labels = tf.cast(labels, dtype=tf.int32)

    true_pos = tf.math.count_nonzero(predicted * labels)
    false_pos = tf.math.count_nonzero(predicted * (labels - 1))
    false_neg = tf.math.count_nonzero((predicted - 1) * labels)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    fmeasure = (2 * precision * recall) / (precision + recall)
    return tf.cast(fmeasure, tf.float32)


class NodeMulticlassTask(GraphTaskModel):
    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters(mp_style)
        these_hypers: Dict[str, Any] = {
            "loss_at_every_layer": False, # whether to apply loss for all layer results, setting this to True will override the option for use_intermediate_gnn_results to True
        }
        super_params.update(these_hypers)
        return super_params

    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None):
        if params['loss_at_every_layer']: # If loss at every layer required, 'use_intermediate_gnn_results' must be True, so it's overridden
            params['use_intermediate_gnn_results'] = True
        self._loss_at_every_layer = params.get("loss_at_every_layer", False)

        super().__init__(params, dataset=dataset, name=name)
        if not hasattr(dataset, "num_node_target_labels"):
            raise ValueError(f"Provided dataset of type {type(dataset)} does not provide num_node_target_labels information.")
        self._num_labels = dataset.num_node_target_labels

    def build(self, input_shapes):
        with tf.name_scope(self.__class__.__name__):
            with tf.name_scope('projection_to_classes'):
                self.node_to_labels_layer = tf.keras.layers.Dense(units=self._num_labels, use_bias=True)
                self.node_to_labels_layer.build((None, self._params["gnn_hidden_dim"]))
        super().build(input_shapes)

    def compute_task_output(
        self, batch_features, final_node_representations, training: bool
    ):
        """ Different returns depending on _use_intermediate_gnn_results,
        if False final_node_representations is tf.Tensor and we return per_node_logits as List[tf.tensor] of len(1)
        if True final_node_representations is Tuple[tf.tensor, List[tf.tensor]] and we return per_node_logits as List[tf.tensor]
        """ 

        if not self._use_intermediate_gnn_results:
            per_node_logits = [self.node_to_labels_layer(final_node_representations)]
        else:
            per_node_logits = [self.node_to_labels_layer(final_node_representations_per_layer) for final_node_representations_per_layer in final_node_representations[1]]
        return (per_node_logits,)

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

    @tf.function(input_signature=(tf.TensorSpec((None, None)), tf.TensorSpec((None, None))))
    def _fast_task_metrics(self, per_node_logits, node_labels):
        per_node_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=per_node_logits, labels=node_labels
        )
        loss = tf.reduce_mean(tf.reduce_sum(per_node_losses, axis=-1))  # Compute mean loss _per node_
        f1_score = micro_f1(per_node_logits, node_labels)

        return loss, f1_score

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
        avg_digit_acc = np.average(r["final_layer_digit_accuracy"] for r in task_results)
        avg_sudoku_acc = np.average(r["final_layer_sudoku_accuracy"] for r in task_results)
        return -avg_digit_acc, f"Avg DigitAccuracy: {avg_digit_acc:.3f}, AvgSudokuAccuracy: {avg_sudoku_acc:.3f}, Avg MicroF1: {avg_microf1:.3f}"
