import numpy as np
import pandas as pd
import seaborn as sns

genome_length_to_filename = {
    5: "./experiment_results/trajectory_2021-06-11T22:12:25.530587.csv",
    10: "./experiment_results/trajectory_2021-06-11T16:28:30.262541.csv",
    15: "./experiment_results/trajectory_2021-06-11T08:00:50.091424.csv",
    20: "./experiment_results/trajectory_2021-06-10T16:00:53.545725.csv",
    25: "./experiment_results/trajectory_2021-06-09T19:19:02.479717.csv"
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
    rho_en_vn_output_filename = f"./experiment_results/plots/rho_en_vn_{genome_length}.pdf"
    rho_en_vm_output_filename = f"./experiment_results/plots/rho_en_vm_{genome_length}.pdf"
    rho_vn_vm_output_filename = f"./experiment_results/plots/rho_vn_vm_{genome_length}.pdf"

    evolution_plot = sns.relplot(kind="line", y=evolution_average_losses, x=range(1, 101), aspect=2.0).set_axis_labels("Number of epochs of training", "Average loss")
    evolution_plot.tight_layout()
    evolution_plot.fig.savefig(evolution_output_filename)

    validation_plot = sns.relplot(kind="line", y=validation_average_losses, x=range(1, 101), aspect=2.0).set_axis_labels("Number of epochs of training", "Average loss")
    axes = validation_plot.axes[0][0]
    min_x = validation_average_losses.argmin() + 1
    min_y = validation_average_losses.min()
    axes.plot(min_x, min_y, 'ro')
    annotation_text = "Minimum: ({min_x}, {min_y:.3f})".format(min_x=min_x, min_y=min_y)
    axes.annotate(annotation_text, (min_x, min_y), xytext=(min_x + 3, min_y + 0.2), arrowprops={'arrowstyle': '->'})
    validation_plot.tight_layout()
    validation_plot.fig.savefig(validation_output_filename)

    evolution_losses_before_overfit = evolution_losses.loc[
       :,
      'Evolution loss after 1 epochs':f"Evolution loss after {min_x} epochs"
    ]
    validation_losses_before_overfit = validation_losses.loc[
       :,
       'Validation loss after 1 epochs':f"Validation loss after {min_x} epochs"
    ]
    rho_en_vn = map(
        lambda n: evolution_losses_before_overfit.iloc[:, n].corr(validation_losses_before_overfit.iloc[:, n]),
        range(0, min_x)
    )
    rho_en_vm = map(
        lambda n: evolution_losses_before_overfit.iloc[:, n].corr(validation_losses_before_overfit.iloc[:, -1]),
        range(0, min_x)
    )
    rho_vn_vm = map(
        lambda n: validation_losses_before_overfit.iloc[:, n].corr(validation_losses_before_overfit.iloc[:, -1]),
        range(0, min_x)
    )

    rho_en_vn_plot = sns.relplot(kind="line", y=list(rho_en_vn), x=range(1, min_x + 1), aspect=2.0).set_axis_labels("Number of epochs of training", "\u03c1(en, vn)")
    rho_en_vn_plot.tight_layout()
    rho_en_vn_plot.fig.savefig(rho_en_vn_output_filename)

    rho_en_vm_plot = sns.relplot(kind="line", y=list(rho_en_vm), x=range(1, min_x + 1), aspect=2.0).set_axis_labels("Number of epochs of training", "\u03c1(en, vm)")
    rho_en_vm_plot.tight_layout()
    rho_en_vm_plot.fig.savefig(rho_en_vm_output_filename)
 
    rho_vn_vm_plot = sns.relplot(kind="line", y=list(rho_vn_vm), x=range(1, min_x + 1), aspect=2.0).set_axis_labels("Number of epochs of training", "\u03c1(vn, vm)")
    rho_vn_vm_plot.tight_layout()
    rho_vn_vm_plot.fig.savefig(rho_vn_vm_output_filename)
