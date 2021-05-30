import csv
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from dagcnn_torch import Population

def normalize_image_data(image_data, full_data):
    return (image_data - full_data.mean()) / full_data.std()

def calculate_loss(individual, criterion, loader):
    losses = []
    for _, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda().flatten()
        predictions = individual(images)
        losses.append(criterion(predictions, labels))
    return(sum(losses) / len(loader))

if __name__ == "__main__":
    n_genomes = 100
    n_epochs = 100

    full_data = torch.load("./datasets/cifar-10/raw/all_training_data.pt").to(dtype=torch.float32)

    evolution_data = torch.load("./datasets/cifar-10/processed/evolution_data.pt").to(dtype=torch.float32)
    evolution_data = normalize_image_data(evolution_data, full_data)
    evolution_labels = torch.load("./datasets/cifar-10/processed/evolution_labels.pt").to(dtype=torch.long)
    evolution_dataset = TensorDataset(evolution_data, evolution_labels)
    evolution_loader = DataLoader(evolution_dataset, shuffle=False, pin_memory=True)

    validation_data = torch.load("./datasets/cifar-10/processed/validation_data.pt").to(dtype=torch.float32)
    validation_data = normalize_image_data(validation_data, full_data)
    validation_labels = torch.load("./datasets/cifar-10/processed/validation_labels.pt").to(dtype=torch.long)
    validation_dataset = TensorDataset(validation_data, validation_labels)
    validation_loader = DataLoader(validation_dataset, shuffle=False, pin_memory=True)

    population = Population.make_random(n_genomes, (3, 32, 32), 10, 1, 15)
    genome_index = 0
    results = []

    for genome in population.genomes:
        result = [repr(genome)]
        evolution_results = []
        validation_results = []

        individual = genome.to_individual()
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(individual.parameters())

        individual.eval()
        evolution_loss = calculate_loss(individual, criterion, evolution_loader)
        validation_loss = calculate_loss(individual, criterion, validation_loader)

        evolution_results.append(evolution_loss.item())
        validation_results.append(validation_loss.item())

        individual.train()
        evolution_loss = calculate_loss(individual, criterion, evolution_loader)
        optimizer.zero_grad()
        evolution_loss.backward()
        optimizer.step()

        for epoch_index in range(n_epochs):
            print(f"[{datetime.now()}] {genome_index}/{epoch_index}")
            individual.eval()
            evolution_loss = calculate_loss(individual, criterion, evolution_loader)
            validation_loss = calculate_loss(individual, criterion, validation_loader)

            evolution_results.append(evolution_loss.item())
            validation_results.append(validation_loss.item())

            individual.train()
            evolution_loss = calculate_loss(individual, criterion, evolution_loader)
            optimizer.zero_grad()
            evolution_loss.backward()
            optimizer.step()

        result += evolution_results
        result += validation_results

        results.append(result)            
        genome_index += 1

    filename = f"./experiment_results/trajectory_{datetime.now().isoformat()}.csv"
    header = ["Genome"]
    header += list(map(lambda n: f"Evolution loss after {n} epochs", range(n_epochs + 1)))
    header += list(map(lambda n: f"Validation loss after {n} epochs", range(n_epochs + 1)))

    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(results)


