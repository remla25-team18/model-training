'''
check_kwarg_dataset_split.py
'''
import ast


class DatasetSplitKwargChecker:
    '''
    Checks whether additional train_test_split function parameters
    (aside from the first two) need to use keyword arguments
    '''
    name = "flake8_dataset_split_kwarg_checker"

    def __init__(self, tree):
        self.tree = tree

    def run(self):
        '''
        Informs the user when additional train_test_split function parameters
        (aside from the first two) are not using keyword arguments
        '''
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                func_name = getattr(node.func, "id", None)

                if func_name == "train_test_split":
                    allowed_positional = 2  # the first two parameters (X, y) don't need keywords
                    num_positional = len(node.args)

                    if num_positional > allowed_positional:
                        # We want to flag when the rest of the parameters don't have keywords (are positional)
                        yield (
                            node.lineno,
                            node.col_offset,
                            "TTS001 Use keyword arguments for parameters after the first two in train_test_split",
                            type(self),
                        )
