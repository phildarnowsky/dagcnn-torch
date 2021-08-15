from datetime import datetime
import torch

def train_once(net, training_loader, criterion, optimizer):
    net.train()
    for training_examples, training_labels in training_loader:
        training_examples = training_examples.cuda()
        training_labels = training_labels.cuda()
        training_predictions = net(training_examples)
 
        training_loss = criterion(training_predictions, training_labels.flatten())
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

def evaluate_on_test_set(net, test_loader):
    n_correct = 0
    n_attempted = 0

    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.flatten().cuda()
            outputs = net(images)
            predictions = torch.argmax(outputs, 1)
            n_correct += (predictions == labels).sum().item()
            n_attempted += predictions.size(0)

    return n_correct / n_attempted

def evaluate_accuracy(genome, n_epochs, checking_interval, training_loader, test_loader):
    net = genome.to_individual()
    accuracies = {0: evaluate_on_test_set(net, test_loader)}

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    for epoch_index in range(1, n_epochs + 1):
        train_once(net, training_loader, criterion, optimizer)
        if epoch_index % checking_interval == 0:
            saynow(f"EPOCH {epoch_index}")
            accuracies[epoch_index] = evaluate_on_test_set(net, test_loader)

    return accuracies

def saynow(text):
    print(f"[{datetime.now()}] {text}") 

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    import pickle
    import sys

    from dagcnn_torch import *

    n_epochs = 50
    checking_interval = 5
    batch_size = 250

    full_training_data = torch.load("./datasets/cifar-10/raw/all_training_data.pt").to(dtype=torch.float32)
    full_training_data_mean = full_training_data.mean()
    full_training_data_std = full_training_data.std()

    training_data = torch.load("./datasets/cifar-10/processed/training_data.pt").to(dtype=torch.float32)
    training_data = (training_data - full_training_data_mean) / full_training_data_std
    training_labels = torch.load("./datasets/cifar-10/processed/training_labels.pt").to(dtype=torch.long)
    training_dataset = TensorDataset(training_data, training_labels)
    training_loader = DataLoader(training_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)

    test_data = torch.load("./datasets/cifar-10/raw/all_test_data.pt").to(dtype=torch.float32)
    test_data = (test_data - full_training_data_mean) / full_training_data_std
    test_labels = torch.load("./datasets/cifar-10/raw/all_test_labels.pt").to(dtype=torch.long)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)


    input_filenames = sys.argv[1:]
    for run_index, input_filename in enumerate(input_filenames):
        with open(input_filename, "rb") as f:
            genome_stats = pickle.load(f)['fitnesses']
        genomes = list(genome_stats.keys())
        accuracy_cache = {}

        for genome_index, genome in enumerate(genomes):
            saynow(f"GENOME {run_index}/{genome_index}/{len(genomes)}")
            cache_key = genome.to_cache_key()
            if cache_key not in accuracy_cache:
                accuracy_cache[cache_key] = evaluate_accuracy(genome, n_epochs, checking_interval, training_loader, test_loader)

        saynow(accuracy_cache)

        dump_filename = f"./experiment_results/cifar_10_classifier_accuracy_{datetime.now().isoformat()}.pickle"
        with open(dump_filename, "wb") as f:
            pickle.dump(accuracy_cache, f)
