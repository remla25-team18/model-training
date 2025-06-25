import os

def count_pattern(filename, pattern):
    count = 0
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                if pattern in line:
                    count += 1
    else:
        print(f"Warning: {filename} does not exist.")
    return count

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    pattern_per_file = {
        "test_data.py": "!ML Test Score, Data",
        "test_infrastructure.py": "!ML Test Score, Infra",
        "test_model.py": "!ML Test Score, Model",
        "test_monitoring.py": "!ML Test Score, Monitor",
    }

    # Every test is done automatically, thus worth 1 point each
    ml_test_scores = {}
    for file, pattern in pattern_per_file.items():
        full_path = os.path.join(base_dir, file)
        ml_test_scores[file] = count_pattern(full_path, pattern)

    # ML Test Score (minimum value of all categories)
    min_count = min(ml_test_scores.values())
    return min_count

if __name__ == "__main__":
    main()
