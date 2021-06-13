import pandas as pd
import seaborn as sns

genome_length_to_filename = {
    5: "./experiment_results/trajectory_2021-06-11T22:12:25.530587.csv",
#   10: "./experiment_results/trajectory_2021-06-11T16:28:30.262541.csv",
#   15: "./experiment_results/trajectory_2021-06-11T08:00:50.091424.csv",
#   20: "./experiment_results/trajectory_2021-06-10T16:00:53.545725.csv",
#   25: "./experiment_results/trajectory_2021-06-09T19:19:02.479717.csv"
}

genome_length_to_data_frame = {length : pd.read_csv(filename) for length, filename in genome_length_to_filename.items()}

for genome_length, df in genome_length_to_data_frame.items():
    evolution_losses = df.loc[
       :,
      'Evolution loss after 1 epochs':'Evolution loss after 100 epochs'
    ]
    validation_losses = df.loc[
       :,
      'Validation loss after 1 epochs':'Validation loss after 100 epochs'
    ]
    evolution_average_losses = evolution_losses.mean(axis=0)
    validation_average_losses = validation_losses.mean(axis=0)

    evolution_output_filename = f"./experiment_results/plots/average_evolution_loss_length_{genome_length}.pdf"
    validation_output_filename = f"./experiment_results/plots/average_validation_loss_length_{genome_length}.pdf"

    evolution_plot = sns.relplot(kind="line", y=evolution_average_losses, x=range(1, 101))
    evolution_plot.fig.savefig(evolution_output_filename)

    validation_plot = sns.relplot(kind="line", y=validation_average_losses, x=range(1, 101))
    validation_plot.fig.savefig(validation_output_filename)
