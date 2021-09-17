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
        saynow(f"EPOCH {epoch_index}")
        train_once(net, training_loader, criterion, optimizer)
        if epoch_index % checking_interval == 0:
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
    batch_size = 64

    full_training_data = torch.load("./datasets/cifar-10/raw/all_training_data.pt").to(dtype=torch.float32)
    full_training_data_mean = full_training_data.mean()
    full_training_data_std = full_training_data.std()

    training_data = (full_training_data - full_training_data_mean) / full_training_data_std
    training_labels = torch.load("./datasets/cifar-10/raw/all_training_labels.pt").to(dtype=torch.long)
    training_dataset = TensorDataset(training_data, training_labels)
    training_loader = DataLoader(training_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)

    test_data = torch.load("./datasets/cifar-10/raw/all_test_data.pt").to(dtype=torch.float32)
    test_data = (test_data - full_training_data_mean) / full_training_data_std
    test_labels = torch.load("./datasets/cifar-10/raw/all_test_labels.pt").to(dtype=torch.long)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)

    population_filenames = [
		'./experiment_results/cifar_10_classifier_2021-09-13T16:05:59.312205.pickle',
		'./experiment_results/cifar_10_classifier_2021-09-14T03:22:33.726833.pickle',
		'./experiment_results/cifar_10_classifier_2021-09-14T03:34:36.733487.pickle',
		'./experiment_results/cifar_10_classifier_2021-09-14T12:15:55.592332.pickle',
		'./experiment_results/cifar_10_classifier_2021-09-14T12:45:32.431689.pickle',
		'./experiment_results/cifar_10_classifier_2021-09-14T23:05:27.655204.pickle',
		'./experiment_results/cifar_10_classifier_2021-09-14T23:43:58.379581.pickle',
		'./experiment_results/cifar_10_classifier_2021-09-15T09:01:41.532084.pickle'
    ]
    accuracy_filenames = [
		'./experiment_results/cifar_10_classifier_accuracy_2021-09-15T09:14:27.384427.pickle',
		'./experiment_results/cifar_10_classifier_accuracy_2021-09-15T20:40:54.115128.pickle',
		'./experiment_results/cifar_10_classifier_accuracy_2021-09-15T21:02:43.688800.pickle',
		'./experiment_results/cifar_10_classifier_accuracy_2021-09-16T09:00:43.647409.pickle',
		'./experiment_results/cifar_10_classifier_accuracy_2021-09-16T09:32:16.449184.pickle',
		'./experiment_results/cifar_10_classifier_accuracy_2021-09-17T00:26:15.461533.pickle',
		'./experiment_results/cifar_10_classifier_accuracy_2021-09-17T01:33:03.914967.pickle',
		'./experiment_results/cifar_10_classifier_accuracy_2021-09-17T12:39:50.590620.pickle'
    ]
   
    results = {}
    for (population_filename, accuracy_filename) in zip(population_filenames, accuracy_filenames):
        with open(population_filename, "rb") as f:
            population_data = pickle.load(f)
        with open(accuracy_filename, "rb") as f:
            accuracy_data = pickle.load(f)
    
        best_accuracy = 0
        best_genome_key = None
        for genome_key in accuracy_data['accuracies']:
            accuracies = accuracy_data['accuracies'][genome_key]
            best_genome_accuracy = max(accuracies.values())
            if best_genome_accuracy > best_accuracy:
                best_genome_key = genome_key
                best_accuracy = best_genome_accuracy
        saynow(population_data['hyperparameters'])
        saynow(best_genome_key)
        saynow(best_accuracy)

        best_genome = None
        for genome in population_data['fitnesses']:
            if genome.to_cache_key() == best_genome_key:
                best_genome = genome
        result = evaluate_accuracy(genome, n_epochs, checking_interval, training_loader, test_loader)
        saynow(result)
        results[best_genome] = result

    dump_filename = f"./experiment_results/experiment_7_bakeoff_{datetime.now().isoformat()}.pickle"
    with open(dump_filename, "wb") as f:
        pickle.dump(results, f)
