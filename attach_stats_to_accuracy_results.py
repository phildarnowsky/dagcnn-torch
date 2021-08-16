import pickle
import torch

classifier_accuracy_filenames = [
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T19:02:08.330579.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T19:18:18.683264.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T19:18:51.397051.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T19:30:24.737644.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T19:30:28.019542.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T19:33:17.009446.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T23:23:14.958888.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T23:27:41.830961.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T23:28:06.970653.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T23:34:53.535286.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T23:35:21.240931.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-15T23:35:43.615475.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T03:16:52.790153.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T03:23:57.262093.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T03:24:10.955506.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T03:29:49.890731.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T03:30:23.534820.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T03:33:59.575763.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T08:07:57.059075.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T08:18:33.862363.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T08:19:01.583471.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T08:22:10.239127.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T08:22:16.883901.pickle",
    "./experiment_results/cifar_10_classifier_accuracy_2021-08-16T08:25:29.821265.pickle"
]

for filename in classifier_accuracy_filenames:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    accuracy_cache = data['accuracies']
    accuracy_hashes = list(accuracy_cache.values())
    accuracy_list = []
    for accuracy_hash in accuracy_hashes:
        for key in accuracy_hash:
            if key != 0:
                accuracy_list.append(accuracy_hash[key])

    accuracies = torch.tensor(accuracy_list)
    n_distinct = len(accuracy_cache)
    max_accuracy = accuracies.max().item()
    avg_accuracy = accuracies.mean().item()
    median_accuracy = accuracies.median().item()
    accuracy_std = accuracies.std().item()
    statistics = {'n_distinct': n_distinct, 'max_accuracy': max_accuracy, 'avg_accuracy': avg_accuracy, 'median_accuracy': median_accuracy, 'accuracy_std': accuracy_std}

    data['statistics'] = statistics
    with open(filename, "wb") as f:
        pickle.dump(data, f)
