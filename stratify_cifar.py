import torch

with open("./datasets/cifar-10/raw/all_training_data.pt", 'rb') as fo:
    data = torch.load(fo)

with open("./datasets/cifar-10/raw/all_training_labels.pt", 'rb') as fo:
    labels = torch.load(fo)

dataset = torch.utils.data.TensorDataset(data, labels)
sampler = torch.utils.data.DataLoader(dataset, shuffle=True)

evolution_counts = [0] * 10
validation_counts = [0] * 10

evolution_rows = torch.tensor([]).reshape(0, 3, 32, 32)
evolution_labels = torch.tensor([]).reshape(0, 1)
validation_rows = torch.tensor([]).reshape(0, 3, 32, 32)
validation_labels = torch.tensor([]).reshape(0, 1)

for batch_index, (image, label) in enumerate(sampler):
    print(batch_index)
    if evolution_counts[label.item()] < 450:
        evolution_counts[label.item()] += 1
        evolution_rows = torch.cat([evolution_rows, image])
        evolution_labels = torch.cat([evolution_labels, label])
    elif validation_counts[label.item()] < 50:
        validation_counts[label.item()] += 1
        validation_rows = torch.cat([validation_rows, image])
        validation_labels = torch.cat([validation_labels, label])

with open("./datasets/cifar-10/processed/evolution_data.pt", 'wb') as fo:
    torch.save(evolution_rows, fo)

with open("./datasets/cifar-10/processed/evolution_labels.pt", 'wb') as fo:
    torch.save(evolution_labels, fo)

with open("./datasets/cifar-10/processed/validation_data.pt", 'wb') as fo:
    torch.save(validation_rows, fo)

with open("./datasets/cifar-10/processed/validation_labels.pt", 'wb') as fo:
    torch.save(validation_labels, fo)
