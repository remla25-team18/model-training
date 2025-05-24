from pylint.lint import PyLinter
from .check_random_state_dataset_split import DatasetSplitRandomStateChecker


def register(linter: PyLinter):
    linter.register_checker(DatasetSplitRandomStateChecker(linter))
