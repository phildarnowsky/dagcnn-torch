def evaluate_accuracy(genome, n_epochs, checking_intervals):
    return []

if __name__ == "__main__":
    import pickle
    import sys

    from dagcnn_torch import *

    n_epochs = 25
    checking_intervals = 5

    input_filename = sys.argv[1]
    with open(input_filename, "rb") as f:
        genome_stats = pickle.load(f)
    genomes = list(genome_stats.keys())
    accuracy_cache = {}

    for genome in genomes:
        cache_key = genome.to_cache_key()
        if cache_key not in accuracy_cache:
            accuracy_cache[cache_key] = evaluate_accuracy(genome, n_epochs, checking_intervals)

    print(accuracy_cache)
