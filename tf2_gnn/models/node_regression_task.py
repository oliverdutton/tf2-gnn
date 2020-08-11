from typing import Any, Dict, Iterable, List, NamedTuple, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf

from tf2_gnn.data import GraphDataset
from tf2_gnn.models import GraphTaskModel

class NodeRegressionTask(GraphTaskModel):
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
            with tf.name_scope('projection_to_outputs'):
                self.node_to_labels_layer = tf.keras.layers.Dense(units=self._num_labels, use_bias=True)
                self.node_to_labels_layer.build((None, self._params["gnn_hidden_dim"]))
        super().build(input_shapes)

    def compute_task_output(
        self, batch_features, final_node_representations, training: bool
    ):
        """ Different returns depending on _use_intermediate_gnn_results,
        if False final_node_representations is tf.Tensor and we return per_node_results as List[tf.tensor] of len(1)
        if True final_node_representations is Tuple[tf.tensor, List[tf.tensor]] and we return per_node_results as List[tf.tensor]
        """ 

        if not self._use_intermediate_gnn_results:
            per_node_results = [self.node_to_labels_layer(final_node_representations)]
        else:
            per_node_results = [self.node_to_labels_layer(final_node_representations_per_layer) for final_node_representations_per_layer in final_node_representations[1]]
        return (per_node_results,)

    def compute_task_metrics(
        self, batch_features, task_output, batch_labels
    ) -> Dict[str, tf.Tensor]:
        (node_predictions,) = task_output
        node_values = batch_labels["node_values"]

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

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        n_nodes = sum(
            batch_task_result["n_nodes"] for batch_task_result in task_results
        ) # total number of nodes calculated in epoch
        epoch_mse = sum(
            batch_task_result["mse"]*batch_task_result["n_nodes"] for batch_task_result in task_results
        ) / n_nodes
        epoch_mae = sum(
            batch_task_result["mae"]*batch_task_result["n_nodes"] for batch_task_result in task_results
        ) / n_nodes
        return epoch_mae.numpy(), f"Mean Absolute Error = {epoch_mae.numpy():.3f}, Mean Squared Error = {epoch_mse.numpy():.3f}"
