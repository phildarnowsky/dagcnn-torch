from datetime import datetime
from pickle import dump

import torch
from torch.utils.data import DataLoader, TensorDataset

from dagcnn_torch.grid_search import GridSearch
from dagcnn_torch.population import Population

def saynow(text):
    print(f"[{datetime.now()}] {text}") 

batch_size = 40

full_training_data = torch.load("./datasets/cifar-10/raw/all_training_data.pt").to(dtype=torch.float32)
full_training_data_mean = full_training_data.mean()
full_training_data_std = full_training_data.std()

training_data = torch.load("./datasets/cifar-10/processed/training_data.pt").to(dtype=torch.float32)
training_data = (training_data - full_training_data_mean) / full_training_data_std
training_labels = torch.load("./datasets/cifar-10/processed/training_labels.pt").to(dtype=torch.long)
training_dataset = TensorDataset(training_data, training_labels)
training_loader = DataLoader(training_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)

validation_data = torch.load("./datasets/cifar-10/processed/validation_data.pt").to(dtype=torch.float32)
validation_data = (validation_data - full_training_data_mean) / full_training_data_std
validation_labels = torch.load("./datasets/cifar-10/processed/validation_labels.pt").to(dtype=torch.long)
validation_dataset = TensorDataset(validation_data, validation_labels)
validation_loader = DataLoader(validation_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)

grid_search = GridSearch(
    [
        [['min_n_genes', 'max_n_genes'], [10, 20], [20, 30]],
        [['elitism_fraction'], [0.2], [0.6]],
        [['mutation_probability'], [0.1], [0.001]],
        [['mean_threshold', 'std_threshold'], [0.1, 0.01], [0.4, 0.04], [1.6, 0.16]]
    ]
)

hyperparameter_combos = grid_search.hyperparameter_combos()
saynow(hyperparameter_combos)
n_hyperparameter_combos = len(hyperparameter_combos)

for (population_index, hyperparameters) in enumerate(hyperparameter_combos):
    population = Population.make_random((3, 32, 32), 10, training_loader, validation_loader, **hyperparameters)

    generation_start_callback = lambda population: saynow(f"POPULATION {population_index} OF {n_hyperparameter_combos}, GENERATION {population.generation_index}")
    population.breed(generation_start_callback)

    saynow("COMPUTING ALL FITNESSES FOR FINAL GENERATION")
    final_fitnesses = population.all_fitnesses()
    saynow("AND DONE!")

    dump_filename = f"./experiment_results/cifar_10_classifier_{datetime.now().isoformat()}.pickle"
    dump_payload = {'fitnesses': final_fitnesses, 'hyperparameters': hyperparameters}

    with open(dump_filename, "wb") as f:
        dump(dump_payload, f)
