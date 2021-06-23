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
    n_batches = 0
    for _, (images, labels) in enumerate(loader):
        n_batches += 1
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
    return running_loss / n_batches

def run_epoch(individual, criterion, training_loader, validation_loader, optimizer):
    with torch.no_grad():
        individual.eval()
        pre_training_training_loss = calculate_loss(individual, criterion, training_loader)
        pre_training_validation_loss = calculate_loss(individual, criterion, validation_loader)

    individual.train()
    _ = calculate_loss(individual, criterion, training_loader, optimizer)
    return (pre_training_training_loss.item(), pre_training_validation_loss.item())

if __name__ == "__main__":
    n_genomes = 100
    n_epochs = 100
    min_genome_length = 30
    max_genome_length = 30
    batch_size = 50

    full_data = torch.load("./datasets/cifar-10/raw/all_training_data.pt").to(dtype=torch.float32)

    training_data = torch.load("./datasets/cifar-10/processed/training_data.pt").to(dtype=torch.float32)
    training_data = normalize_image_data(training_data, full_data)
    training_labels = torch.load("./datasets/cifar-10/processed/training_labels.pt").to(dtype=torch.long)
    training_dataset = TensorDataset(training_data, training_labels)
    training_loader = DataLoader(training_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)

    validation_data = torch.load("./datasets/cifar-10/processed/validation_data.pt").to(dtype=torch.float32)
    validation_data = normalize_image_data(validation_data, full_data)
    validation_labels = torch.load("./datasets/cifar-10/processed/validation_labels.pt").to(dtype=torch.long)
    validation_dataset = TensorDataset(validation_data, validation_labels)
    validation_loader = DataLoader(validation_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)

    population = Population.make_random(n_genomes, (3, 32, 32), 10, min_genome_length, max_genome_length)
    genome_index = 0
    results = []

    for genome in population.genomes:
        result = [repr(genome)]

        individual = genome.to_individual()
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(individual.parameters())

        saynow(f"CALCULATING PRETRAINING LOSS FOR GENOME {genome_index}")
        (training_loss, validation_loss) = run_epoch(individual, criterion, training_loader, validation_loader, optimizer)
        training_results = [training_loss]
        validation_results = [validation_loss]
        saynow(training_loss)

        for epoch_index in range(n_epochs):
            saynow(f"{genome_index}/{epoch_index}")
            (training_loss, validation_loss) = run_epoch(individual, criterion, training_loader, validation_loader, optimizer)
            training_results.append(training_loss)
            validation_results.append(validation_loss)
            saynow(training_loss)

        result += training_results
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


