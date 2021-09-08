import math
import os
import pickle
import re
import sys

input_filenames = sys.argv[1:]
genomes = []
accuracies = {}
results = {'A': 0, 'C': 0, 'D': 0, 'K': 0, 'M': 0, 'S': 0}

for filename in input_filenames:
    with open(filename, "rb") as f:
        file_accuracies = pickle.load(f, encoding='bytes')['accuracies']
    for genome in file_accuracies:
        genomes.append(genome)
        accuracies[genome] = max(file_accuracies[genome].values())

genomes = sorted(genomes, key=lambda x: -accuracies[x])
n_genomes_to_keep = math.floor(len(genomes) * 0.1)

print(n_genomes_to_keep)
for genome in genomes[:n_genomes_to_keep]:
    genes = re.split('(?<=\))', genome)
    genes.pop()
    for gene in genes:
        gene = re.sub('[\(\)]', "", gene)
        operation = re.split('_', gene)[0]
        results[operation] += 1

total_nodes = sum(results.values())

for operation in sorted(results):
    n_nodes = results[operation]
    percentage = n_nodes * 1.0 / total_nodes
    print(f"{operation}: {percentage:.4f} {n_nodes}")
print(total_nodes)

print(results)
