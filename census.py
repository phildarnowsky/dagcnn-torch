import os
import pickle
import re
import sys

input_filenames = sys.argv[1:]
genomes = []
results = {}

for filename in input_filenames:
    with open(filename, "rb") as f:
        file_genomes = list(pickle.load(f, encoding='bytes')['accuracies'].keys())
        print(file_genomes)
    genomes = genomes + file_genomes

for genome in genomes:
    genes = re.split('(?<=\))', genome)
    genes.pop()
    for gene in genes:
        gene = re.sub('[\(\)]', "", gene)
        operation = re.split('_', gene)[0]
        if operation not in results.keys():
            results[operation] = 0
        results[operation] += 1

print(results)
