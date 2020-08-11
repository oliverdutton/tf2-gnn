"""Implementation of protein sequence secondary structure dataset"""
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Set
from functools import reduce

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from .graph_dataset import DataFold, GraphSample, GraphBatchTFDataDescription, GraphDataset
from .utils import compute_number_of_edge_types, get_tied_edge_types, process_adjacency_lists

logger = logging.getLogger(__name__)

class PSSGraphSample(GraphSample):
    """
    Data structure holding a protein graph.
    Nodes for each amino acid, node_values of 
    the secondary structure attributed
    """

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_incoming_edges: np.ndarray,
        node_features: np.ndarray,
        node_values: np.ndarray,
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_incoming_edges, node_features)
        self._node_values = node_values

    @property
    def node_values(self) -> float:
        """Target secondary structure value"""
        return self._node_values

    def __str__(self):
        return (
            f"Adj:            {self._adjacency_lists}\n"
            f"Node_features:  {self._node_features}\n"
            f"node_values:  {self._node_values}"
        )

class PSSDataset(GraphDataset[PSSGraphSample]):
    """
    Protein secondary structure Dataset class.

    connectivity is up to how many amino acids away a particular
        amino acid each amino acid has an edge to
    """

    str_to_int = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
    }

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "connectivity":5,
            "max_nodes_per_batch": 2700,
            "add_self_loop_edges": True,
            "tie_fwd_bkwd_edges": True,
        }

    def __init__(self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        logger.info("Initialising Protein secondary structure Dataset.")
        super().__init__(params, metadata=metadata)
        self._params = params
        self._num_fwd_edge_types = self._params['connectivity']

        self._tied_fwd_bkwd_edge_types = get_tied_edge_types(
            tie_fwd_bkwd_edges=params["tie_fwd_bkwd_edges"],
            num_fwd_edge_types=self._num_fwd_edge_types,
        )

        self._num_edge_types = compute_number_of_edge_types(
            tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
            num_fwd_edge_types=self._num_fwd_edge_types,
            add_self_loop_edges=params["add_self_loop_edges"],
        )

        self._node_feature_shape = None
        self._loaded_data: Dict[DataFold, List[PSSGraphSample]] = {}
        logger.debug("Done initialising protein secondary structure dataset.")

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    # -------------------- Data Loading --------------------
    @classmethod
    def default_data_directory(cls):
        curr_dir = Path(os.path.abspath(inspect.getsourcefile(lambda: 0)))
        data_directory = os.path.join(curr_dir.parent.parent, "data")
        return data_directory

    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        """ 
        Returns list of PSSGraphSamples to the dict self._loaded_data with 
            DataFold.(TRAIN/VALIDATION/TEST) as key
        """
        """Load the data from disk."""
        if path is None:
            path = RichPath.create(self.default_data_directory())
        logger.info("Starting to load data.")

        # If we haven't defined what folds to load, load all:
        if folds_to_load is None:
            folds_to_load = {DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST}

        if DataFold.TRAIN in folds_to_load:
            self._loaded_data[DataFold.TRAIN] = self.__load_data(path.join("train.json"))
            logger.debug("Done loading training data.")
        if DataFold.VALIDATION in folds_to_load:
            self._loaded_data[DataFold.VALIDATION] = self.__load_data(path.join("valid.json"))
            logger.debug("Done loading validation data.")
        if DataFold.TEST in folds_to_load:
            self._loaded_data[DataFold.TEST] = self.__load_data(path.join("test.json"))
            logger.debug("Done loading test data.")

    def load_data_from_list(
        self, datapoints: List[Dict[str, Any]], target_fold: DataFold = DataFold.TEST
    ):
        raise NotImplementedError()

    def __load_data(self, data_file: RichPath) -> List[PSSGraphSample]:
        def parse(sequence: str) -> tf.Tensor:
            """ Returns one_hot encoding of sequence
            sequence: str of length n
            output: tf.Tensor of dim (n,20) """
            int_representation = [self.str_to_int[letter] for letter in sequence]
            return tf.one_hot(int_representation, 20)

        data = data_file.read_by_file_suffix()
        # .json expected which is read as a dict with keys 'X' and 'Y' 
        encoded = [(parse(sequence), secondary_structure) for sequence, secondary_structure in zip(data['X'], data['Y'])]

        return self.__process_raw_graphs(encoded)

    def __process_raw_graphs(self, raw_data: Iterable[Any]) -> List[PSSGraphSample]:
        processed_graphs = []
        for node_features, node_values in raw_data:
            (type_to_adjacency_list, type_to_num_incoming_edges) = self.__sequence_to_adjacency_lists(
                num_nodes=len(node_values)
            )
            processed_graphs.append(
                PSSGraphSample(
                    adjacency_lists=type_to_adjacency_list,
                    type_to_node_to_num_incoming_edges=type_to_num_incoming_edges,
                    node_features=node_features,
                    node_values=node_values,
                )
            )
        return processed_graphs

    def __sequence_to_adjacency_lists(self, num_nodes: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """ Generates graph for each protein sequence, based off of datasets 
        connectivity option"""
        connectivity = self._params['connectivity']

        graph_list = [list(zip(range(num_nodes), [i-1]*num_nodes,range(i,num_nodes))) for i in range(1,connectivity+1)]
        graph = reduce(lambda x,y:x+y, graph_list)

        return self.__graph_to_adjacency_lists(
                    graph, num_nodes=num_nodes
                )

    def __graph_to_adjacency_lists(
        self, graph: Iterable[Tuple[int, int, int]], num_nodes: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        raw_adjacency_lists = [[] for _ in range(self.num_edge_types)]

        for src, edge_type, dest in graph:
            raw_adjacency_lists[edge_type].append((src, dest))

        return process_adjacency_lists(
            adjacency_lists=raw_adjacency_lists,
            num_nodes=num_nodes,
            add_self_loop_edges=self.params["add_self_loop_edges"],
            tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
        )

    @property
    def num_node_target_labels(self) -> int:
        return 1

    @property
    def node_feature_shape(self) -> Tuple:
        some_data_fold = next(iter(self._loaded_data.values()))
        return (some_data_fold[0].node_features.shape[-1],)

    # -------------------- Minibatching -------------------- from PPIDataset
    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        data_description = super().get_batch_tf_data_description()
        return GraphBatchTFDataDescription(
            batch_features_types=data_description.batch_features_types,
            batch_features_shapes=data_description.batch_features_shapes,
            batch_labels_types={**data_description.batch_labels_types, "node_values": tf.float32},
            batch_labels_shapes={**data_description.batch_labels_shapes, "node_values": (None,)},
        )

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[PSSGraphSample]:
        loaded_data = self._loaded_data[data_fold]
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(loaded_data)
        return iter(loaded_data)

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["node_values"] = []
        return new_batch

    def _add_graph_to_batch(self, raw_batch, graph_sample: PSSGraphSample) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["node_values"].append(graph_sample.node_values)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_values = super()._finalise_batch(raw_batch)
        batch_values["node_values"] = np.concatenate(raw_batch["node_values"], axis=0)
        return batch_features, batch_values


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
