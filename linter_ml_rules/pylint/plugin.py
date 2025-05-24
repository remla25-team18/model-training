'''
check_kwarg_dataset_split.py
'''
from pylint.lint import PyLinter
from .check_random_state_dataset_split import DatasetSplitRandomStateChecker


def register(linter: PyLinter):
    '''
    Makes the plugin discoverable
    '''
    linter.register_checker(DatasetSplitRandomStateChecker(linter))
