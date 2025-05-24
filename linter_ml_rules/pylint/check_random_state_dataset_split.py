'''
check_random_state_dataset_split.py
'''
from pylint.checkers.base_checker import BaseChecker
import astroid

class DatasetSplitRandomStateChecker(BaseChecker):
    '''
    Checks if the train_test_split has a random_state set
    '''
    name = "dataset-split-random-state"
    priority = -1 # lower priority

    msgs = {
        "W9001": (
            "train_test_split is used without random_state set",
            "train-test-split-missing-random-state",
            "train_test_split should have a random_state for reproducibility.",
        ),
    }

    def visit_call(self, node: astroid.nodes.Call):
        '''
        Informs the user when a random_state is not set in train_test_split
        '''
        if isinstance(node.func, astroid.nodes.Name) and node.func.name == "train_test_split":
            keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
            if "random_state" not in keywords:
                self.add_message("train-test-split-missing-random-state", node=node)
