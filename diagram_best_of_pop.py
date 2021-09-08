import os
import pickle
import re
import sys

output_file_prefix = sys.argv[1]
input_filenames = sys.argv[2:]

results = []
for filename in input_filenames:
    with open(filename, "rb") as f:
        results.append(pickle.load(f))

by_max_accuracy = sorted(results, key=lambda x: -x['statistics']['max_accuracy'])
best_pop = by_max_accuracy[0]

best_accuracy = 0
best_genome = None
for genome in best_pop['accuracies']:
    accuracies = best_pop['accuracies'][genome]
    best_genome_accuracy = max(accuracies.values())
    if best_genome_accuracy > best_accuracy:
        best_genome = genome
        best_accuracy = best_genome_accuracy

print(best_genome)
print(best_accuracy)

genes = re.split('(?<=\))', best_genome)
genes.pop()

nodes = []
edges = []
output_indices = list(map(lambda x: str(x), range(0, len(genes))))

for (gene_index, gene) in enumerate(genes):
    gene = re.sub('[\(\)]', "", gene)
    [operation, parameters, inputs] = re.split('_', gene)
    inputs = re.split(',', inputs)
    parameters = re.sub(',', '/', parameters)
    if operation == "A":
        label = "AvgPool"
    if operation == "C":
        label = f"Conv {parameters}"
    if operation == "D":
        label = f"DepSep {parameters}"
    if operation == "M":
        label = "MaxPool"
    if operation == "K":
        label = "Cat"
    if operation == "S":
        label = "+"

    nodes.append(f"{gene_index} [label=\"{label}\"]")
    for input in inputs:
        if input == "-1":
            edges.append(f"model_input -> {gene_index}")
        else:
            edges.append(f"{input} -> {gene_index}")
            if input in output_indices:
                output_indices.remove(input)

for output_index in output_indices:
    edges.append(f"{output_index} -> gsum")

node_lines = "\n".join(nodes)
edge_lines = "\n".join(edges)

output_dot = f"""
strict digraph {{
model_input [label="Model input"]
gsum [label="+"]
gap [label="Global average pooling"]
fc [label="Fully connected layer"]
model_output [label="Model output"]
gsum -> gap
gap -> fc
fc -> model_output

{node_lines}
{edge_lines}
}}
"""

dot_filename = f"{output_file_prefix}.dot"
pdf_filename = f"{output_file_prefix}.pdf"
with open(dot_filename, "w") as f:
    f.write(output_dot)
os.system(f"dot -Tpdf {dot_filename} >{pdf_filename}")
