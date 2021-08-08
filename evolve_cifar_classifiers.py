from datetime import datetime
from pickle import dump

import torch
from torch.utils.data import DataLoader, TensorDataset

from dagcnn_torch.population import Population

def saynow(text):
    print(f"[{datetime.now()}] {text}") 

batch_size = 10
n_genomes = 4
min_n_genes = 3
max_n_genes = 5
n_generations = 10
elitism_fraction = 0.2
mutation_probability = 1.003

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

population = Population.make_random(n_genomes, (3, 32, 32), 10, min_n_genes, max_n_genes, training_loader, validation_loader, elitism_fraction=elitism_fraction, mutation_probability=mutation_probability)

saynow(list(map(lambda genome: genome.to_cache_key(), population._genomes)))
for i in range(n_generations):
    saynow(f"GENERATION {i}")
    population.breed_next_generation()
saynow(list(map(lambda genome: genome.to_cache_key(), population._genomes)))

saynow("COMPUTING ALL FITNESSES FOR FINAL GENERATION")
final_fitnesses = population.all_fitnesses()
saynow("AND DONE!")

dump_filename = f"./experiment_results/cifar_10_classifier_{datetime.now().isoformat()}.pickle"
with open(dump_filename, "wb") as f:
    dump(final_fitnesses, f)

