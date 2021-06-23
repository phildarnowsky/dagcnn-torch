import torch

with open("./datasets/cifar-10/raw/all_training_data.pt", 'rb') as fo:
    data = torch.load(fo)

with open("./datasets/cifar-10/raw/all_training_labels.pt", 'rb') as fo:
    labels = torch.load(fo)

dataset = torch.utils.data.TensorDataset(data, labels)
sampler = torch.utils.data.DataLoader(dataset, shuffle=True)

training_counts = [0] * 10
validation_counts = [0] * 10

training_rows = torch.tensor([]).reshape(0, 3, 32, 32)
training_labels = torch.tensor([]).reshape(0, 1)
validation_rows = torch.tensor([]).reshape(0, 3, 32, 32)
validation_labels = torch.tensor([]).reshape(0, 1)

for batch_index, (image, label) in enumerate(sampler):
    print(batch_index)
    if training_counts[label.item()] < 450:
        training_counts[label.item()] += 1
        training_rows = torch.cat([training_rows, image])
        training_labels = torch.cat([training_labels, label])
    elif validation_counts[label.item()] < 50:
        validation_counts[label.item()] += 1
        validation_rows = torch.cat([validation_rows, image])
        validation_labels = torch.cat([validation_labels, label])

with open("./datasets/cifar-10/processed/training_data.pt", 'wb') as fo:
    torch.save(training_rows, fo)

with open("./datasets/cifar-10/processed/training_labels.pt", 'wb') as fo:
    torch.save(training_labels, fo)

with open("./datasets/cifar-10/processed/validation_data.pt", 'wb') as fo:
    torch.save(validation_rows, fo)

with open("./datasets/cifar-10/processed/validation_labels.pt", 'wb') as fo:
    torch.save(validation_labels, fo)
