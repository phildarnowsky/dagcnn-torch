import csv
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from dagcnn_torch import Population

def saynow(text):
    print(f"[{datetime.now()}] {text}") 

def normalize_image_data(image_data, full_data):
    return (image_data - full_data.mean()) / full_data.std()

def calculate_loss(individual, criterion, loader, optimizer=None):
    running_loss = None
    n_images = 0
    for _, (images, labels) in enumerate(loader):
        n_images += 1
        images = images.cuda()
        labels = labels.cuda().flatten()
        predictions = individual(images)
        loss = criterion(predictions, labels)
        if running_loss:
            running_loss += loss
        else:
            running_loss = loss.detach()

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return running_loss / n_images

def run_epoch(individual, criterion, evolution_loader, validation_loader, optimizer):
    with torch.no_grad():
        individual.eval()
        pre_training_evolution_loss = calculate_loss(individual, criterion, evolution_loader)
        pre_training_validation_loss = calculate_loss(individual, criterion, validation_loader)

    individual.train()
    _ = calculate_loss(individual, criterion, evolution_loader, optimizer)
    return (pre_training_evolution_loss.item(), pre_training_validation_loss.item())

if __name__ == "__main__":
    n_genomes = 2
    n_epochs = 2
    genome_length = 10

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

    population = Population.make_random(n_genomes, (3, 32, 32), 10, 50, 50)
    genome_index = 0
    results = []

    for genome in population.genomes:
        result = [repr(genome)]

        individual = genome.to_individual()
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(individual.parameters())

        saynow(f"CALCULATING PRETRAINING LOSS FOR GENOME {genome_index}")
        (evolution_loss, validation_loss) = run_epoch(individual, criterion, evolution_loader, validation_loader, optimizer)
        evolution_results = [evolution_loss]
        validation_results = [validation_loss]
        saynow(evolution_loss)

        for epoch_index in range(n_epochs):
            saynow(f"{genome_index}/{epoch_index}")
            (evolution_loss, validation_loss) = run_epoch(individual, criterion, evolution_loader, validation_loader, optimizer)
            evolution_results.append(evolution_loss)
            validation_results.append(validation_loss)
            saynow(evolution_loss)

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


