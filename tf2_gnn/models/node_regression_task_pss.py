from typing import Any, Dict, Iterable, List, NamedTuple, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf

from tf2_gnn.data import GraphDataset
from tf2_gnn.models import NodeRegressionTask

class PSSTask(NodeRegressionTask):
    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None):
        super().__init__(params, dataset=dataset, name=name)
     
    def compute_task_metrics(
        self, batch_features, task_output, batch_labels
    ) -> Dict[str, tf.Tensor]:
        (node_predictions,) = task_output
        node_values = batch_labels["node_values"]

        """ 
        Extra section to mask missing node_values in experimental so they don't contribute
            to losses
        """
        mask = np.not_equal(node_values,0)
        node_values, node_predictions = node_values[mask], tf.boolean_mask(node_predictions, mask, axis=1)

        mse = tf.reduce_mean(
            tf.losses.mean_squared_error(node_values, node_predictions[-1])
        )
        mae = tf.reduce_mean(
            tf.losses.mean_absolute_error(node_values, node_predictions[-1])
        )
        n_nodes = node_values.shape[0]
        task_metrics = {
            "loss": mse,
            "mse": mse,
            "mae": mae,
            "n_nodes": n_nodes
        }

        if self._loss_at_every_layer:
            """ If _loss_at_every_layer the f1_score for the final layer is returned but loss is calculated 
                    as the mean loss of each layer"""          
            every_layer_mse = tf.reduce_mean([
                tf.reduce_mean(
                    tf.losses.mean_squared_error(node_predictions_per_layer, node_values)
                ) for node_predictions_per_layer in node_predictions
            ])
            task_metrics.update({"loss":every_layer_mse, "final_layer_loss":mse})
        return task_metrics