"""Implementation of sudoku dataset"""
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Set
import csv

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from .graph_dataset import DataFold, GraphSample, GraphBatchTFDataDescription, GraphDataset
from .utils import compute_number_of_edge_types, get_tied_edge_types, process_adjacency_lists

logger = logging.getLogger(__name__)


def sudoku_edges(bidirectional=True):
    """ Gets all the edges between sudoku boxes in a regular 9x9 puzzle. 
    Indexes are 0 to 80
    
    Returns list of tuples
    Tuples are in both directions i.e. both (0,5) and (5,0) in list
    """

    def cross(a):
        return [(i, j) for i in a.flatten() for j in a.flatten() if not i == j]

    idx = np.arange(81).reshape(9, 9)
    rows, columns, squares = [], [], []
    for i in range(9):
        rows += cross(idx[i, :])
        columns += cross(idx[:, i])
    for i in range(3):
        for j in range(3):
            squares += cross(idx[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])
    ordered_pairs = list(set(rows + columns + squares))
    unordered_pairs = [(a,b) for a,b in ordered_pairs if a>b]
    
    if bidirectional:
        return unordered_pairs
    else:
        return ordered_pairs

class SudokuGraphSample(GraphSample):
    """
    Data structure holding a single Sudoku graph.  
    """

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_incoming_edges: np.ndarray,
        node_features: np.ndarray,
        node_labels: np.ndarray,
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_incoming_edges, node_features)
        self._node_labels = node_labels

    @property
    def node_labels(self) -> float:
        """Target class, the sudoku output value"""
        return self._node_labels

    def __str__(self):
        return (
            f"Adj:            {self._adjacency_lists}\n"
            f"Node_features:  {self._node_features}\n"
            f"node_labels:  {self._node_labels}"
        )

class SudokuDataset(GraphDataset[SudokuGraphSample]):
    """
    Sudoku Dataset class.
    """
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "max_nodes_per_batch": 2700,
            "add_self_loop_edges": True,
            "tie_fwd_bkwd_edges": True,
        }

    def __init__(self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        logger.info("Initialising Sudoku Dataset.")
        super().__init__(params, metadata=metadata)
        self._params = params
        self._num_fwd_edge_types = 1

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
        self._loaded_data: Dict[DataFold, List[SudokuGraphSample]] = {}

        self.raw_adjacency_list = [sudoku_edges()]
        (self.type_to_adjacency_list, self.type_to_num_incoming_edges) = process_adjacency_lists(
            adjacency_lists=self.raw_adjacency_list,
            num_nodes=81,
            add_self_loop_edges=self.params["add_self_loop_edges"],
            tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
        )
        logger.debug("Done initialising Sudoku dataset.")

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
        Returns list of SudokuGraphSamples to the dict self._loaded_data with 
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
            self._loaded_data[DataFold.TRAIN] = self.__load_data(path.join("train.csv"))
            logger.debug("Done loading training data.")
        if DataFold.VALIDATION in folds_to_load:
            self._loaded_data[DataFold.VALIDATION] = self.__load_data(path.join("valid.csv"))
            logger.debug("Done loading validation data.")
        if DataFold.TEST in folds_to_load:
            self._loaded_data[DataFold.TEST] = self.__load_data(path.join("test.csv"))
            logger.debug("Done loading test data.")

    def load_data_from_list(
        self, datapoints: List[Dict[str, Any]], target_fold: DataFold = DataFold.TEST
    ):
        raise NotImplementedError()

    def __load_data(self, data_file: RichPath) -> List[SudokuGraphSample]:
        def read_csv(data_file):
            with open(data_file) as f:
                reader = csv.reader(f, delimiter=',')
                data = [(q, a) for q, a in reader]
                return np.asarray(data)

        def parse(x):
            return list(map(int, list(x)))

        data = read_csv(data_file.__str__())
        encoded = [(parse(q), parse(a)) for q, a in data]

        return self.__process_raw_graphs(encoded)

    def __process_raw_graphs(self, raw_data: Iterable[Any]) -> List[SudokuGraphSample]:

        processed_graphs = [SudokuGraphSample(
            adjacency_lists=self.type_to_adjacency_list,
            type_to_node_to_num_incoming_edges=self.type_to_num_incoming_edges,
            node_features=tf.one_hot(q, 10).numpy(),
            node_labels=tf.one_hot(a, 10)[:,1:].numpy(),
        ) for q, a in raw_data]
        
        return processed_graphs

    @property
    def num_node_target_labels(self) -> int:
        return 9

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
            batch_labels_types={**data_description.batch_labels_types, "node_labels": tf.float32},
            batch_labels_shapes={**data_description.batch_labels_shapes, "node_labels": (None, None)},
        )

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[SudokuGraphSample]:
        loaded_data = self._loaded_data[data_fold]
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(loaded_data)
        return iter(loaded_data)

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["node_labels"] = []
        return new_batch

    def _add_graph_to_batch(self, raw_batch, graph_sample: SudokuGraphSample) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["node_labels"].append(graph_sample.node_labels)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_labels = super()._finalise_batch(raw_batch)
        batch_labels["node_labels"] = np.concatenate(raw_batch["node_labels"], axis=0)
        return batch_features, batch_labels


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
