import numpy as np
import pickle
import sys

def fp(num):
    return np.format_float_positional(num, trim='-')

output_filename = sys.argv[1]
input_filenames = sys.argv[2:]

results = []
for filename in input_filenames:
    with open(filename, "rb") as f:
        results.append(pickle.load(f))

by_max_accuracy = sorted(results, key=lambda x: -x['statistics']['max_accuracy'])
table_lines = list(
    map(
        lambda x: "{max_accuracy:.3f} & {min_n_genes} & {max_n_genes} & {elitism_fraction} & {mutation_probability} & {mean_threshold} & {std_threshold} \\\\ ".format(max_accuracy=x['statistics']['max_accuracy'], min_n_genes=x['hyperparameters']['min_n_genes'], max_n_genes=x['hyperparameters']['max_n_genes'], elitism_fraction=fp(x['hyperparameters']['elitism_fraction']), mutation_probability=fp(x['hyperparameters']['mutation_probability']), mean_threshold=fp(x['hyperparameters']['mean_threshold']), std_threshold=fp(x['hyperparameters']['std_threshold'])),
        by_max_accuracy
    )
)
table_lines = " \hline\n".join(table_lines)

# {max_accuracy:.3f} & {min_n_genes} & {max_n_genes} & {elitism_fraction} & {mutation_probability} & {mean_threshold} & {std_threshold}
table_code = f"""
\\begin{{tabularx}}{{\\textwidth}}{{ | X || X | X | X | X | X | X | }}
\\hline
Maximum accuracy & Minimum \\# genes & Maximum \# genes & Elitism fraction & Mutation probability & Mean threshold & Standard deviation threshold \\\\
\\hline
\\hline
{table_lines}
\\hline
\\end{{tabularx}}
"""

with open(output_filename, "w") as f:
    f.write(table_code)
