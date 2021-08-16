import pickle

evolved_classifier_filenames = [
    "./experiment_results/cifar_10_classifier_2021-08-10T20:33:56.878493.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-10T22:16:52.725532.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-10T22:50:14.424005.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-11T01:52:01.336648.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-11T02:21:32.077561.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-11T02:46:38.582503.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-12T00:06:58.867107.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-12T01:02:54.595239.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-12T01:29:13.036941.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-12T03:25:37.924584.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-12T03:44:26.869042.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-12T03:58:43.708018.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-13T08:01:59.713323.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-13T10:01:59.581979.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-13T11:06:47.127742.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-13T21:49:38.374141.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-13T22:42:21.955981.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-13T23:35:14.937543.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-14T23:31:36.072547.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-15T01:25:57.819410.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-15T02:30:52.599139.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-15T05:32:34.083179.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-15T06:15:12.227498.pickle",
    "./experiment_results/cifar_10_classifier_2021-08-15T06:52:04.514080.pickle"
]

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

for classifier_filename, accuracy_filename in zip(evolved_classifier_filenames, classifier_accuracy_filenames):
    with open(classifier_filename, "rb") as f:
        hyperparameters = pickle.load(f)['hyperparameters']
    with open(accuracy_filename, "rb") as f:
        accuracies = pickle.load(f)
    result = {'hyperparameters': hyperparameters, 'accuracies': accuracies}
    with open(accuracy_filename, "wb") as f:
        pickle.dump(result, f)
