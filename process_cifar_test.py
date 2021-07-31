import pickle

import torch

def unpickle(filename):
    with open(filename, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
    return dict

if __name__ == "__main__":
    filenames = ["test_batch"]

    data = None
    labels = None

    for filename in filenames:
        in_path = f"/home/phil/datasets/cifar-10-batches-py/{filename}"
        file_contents = unpickle(in_path)
        file_data = file_contents[b'data']
        file_labels = file_contents[b'labels']
        file_data = torch.tensor(file_data).reshape((10000, 3, 32, 32))
        if data == None:
            data = file_data
            labels = file_labels
        else:
            data = torch.cat([data, file_data])
            labels = labels + file_labels

    labels = torch.tensor(labels).reshape(10000, 1)

    with open("./datasets/cifar-10/raw/all_test_data.pt", 'wb') as file:
        torch.save(data, file)

    with open("./datasets/cifar-10/raw/all_test_labels.pt", 'wb') as file:
        torch.save(labels, file)

