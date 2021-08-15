import numpy as np
import pandas as pd
import seaborn as sns

genome_length_to_filename_run_1 = {
    5: "./experiment_results/tendency_2021-06-11T22:12:25.530587.csv",
    10: "./experiment_results/tendency_2021-06-11T16:28:30.262541.csv",
    15: "./experiment_results/tendency_2021-06-11T08:00:50.091424.csv",
    20: "./experiment_results/tendency_2021-06-10T16:00:53.545725.csv",
    25: "./experiment_results/tendency_2021-06-09T19:19:02.479717.csv",
    30: "./experiment_results/tendency_2021-06-19T21:02:16.394366.csv"
}

genome_length_to_filename_run_2 = {
    5: "./experiment_results/trajectory_2021-08-13T22:41:31.307840.csv",
    10: "./experiment_results/trajectory_2021-08-14T01:27:43.063621.csv",
    15: "./experiment_results/trajectory_2021-08-14T05:37:39.141266.csv",
    20: "./experiment_results/trajectory_2021-08-14T11:37:55.900189.csv",
    25: "./experiment_results/trajectory_2021-08-15T01:36:18.953254.csv",
    30: "./experiment_results/trajectory_2021-08-15T09:35:36.955481.csv"
}

for (run_index, genome_length_to_filename) in enumerate([genome_length_to_filename_run_1, genome_length_to_filename_run_2]):
    genome_lengths = list(genome_length_to_filename.keys())
    genome_lengths.sort()

    genome_length_to_data_frame = {length : pd.read_csv(filename) for length, filename in genome_length_to_filename.items()}

    training_average_loss_result = pd.DataFrame()
    validation_average_loss_result = pd.DataFrame()
    rho_tn_vn_result = pd.DataFrame()
    rho_tn_vm_result = pd.DataFrame()
    rho_vn_vm_result = pd.DataFrame()

    minima = {}
    max_m = 0

    for genome_length, df in genome_length_to_data_frame.items():
        training_losses = df.loc[
           :,
          'Evolution loss after 1 epochs':'Evolution loss after 100 epochs'
        ]
        validation_losses = df.loc[
           :,
          'Validation loss after 1 epochs':'Validation loss after 100 epochs'
        ]
        training_average_losses = training_losses.mean(axis=0)
        training_df = pd.DataFrame({"loss": training_average_losses, "n_epochs": range(1, 101), "genome_length": genome_length})
        training_average_loss_result = pd.concat([training_average_loss_result, training_df])

        validation_average_losses = validation_losses.mean(axis=0)
        min_x = validation_average_losses.argmin() + 1
        min_y = validation_average_losses.min()
        if min_x > max_m:
            max_m = min_x
        minima[genome_length] = (min_x, min_y)
        validation_df = pd.DataFrame({"loss": validation_average_losses, "n_epochs": range(1, 101), "genome_length": genome_length})
        validation_average_loss_result = pd.concat([validation_average_loss_result, validation_df])

        training_losses_before_overfit = training_losses.loc[
           :,
          'Evolution loss after 1 epochs':f"Evolution loss after {min_x} epochs"
        ]
        validation_losses_before_overfit = validation_losses.loc[
           :,
           'Validation loss after 1 epochs':f"Validation loss after {min_x} epochs"
        ]

        rho_tn_vn = map(
            lambda n: training_losses_before_overfit.iloc[:, n].corr(validation_losses_before_overfit.iloc[:, n]),
            range(0, min_x)
        )
        rho_tn_vn_df = pd.DataFrame({"rho": rho_tn_vn, "n_epochs": range(1, min_x + 1), "genome_length": genome_length})
        rho_tn_vn_result = pd.concat([rho_tn_vn_result, rho_tn_vn_df])

        rho_tn_vm = map(
            lambda n: training_losses_before_overfit.iloc[:, n].corr(validation_losses_before_overfit.iloc[:, -1]),
            range(0, min_x)
        )
        rho_tn_vm_df = pd.DataFrame({"rho": rho_tn_vm, "n_epochs": range(1, min_x + 1), "genome_length": genome_length})
        rho_tn_vm_result = pd.concat([rho_tn_vm_result, rho_tn_vm_df])

        rho_vn_vm = map(
            lambda n: validation_losses_before_overfit.iloc[:, n].corr(validation_losses_before_overfit.iloc[:, -1]),
            range(0, min_x)
        )
        rho_vn_vm_df = pd.DataFrame({"rho": rho_vn_vm, "n_epochs": range(1, min_x + 1), "genome_length": genome_length})
        rho_vn_vm_result = pd.concat([rho_vn_vm_result, rho_vn_vm_df])

    training_plot = sns.relplot(kind="line", data=training_average_loss_result, x="n_epochs", y="loss", hue="genome_length").set_axis_labels("Number of epochs of training", "Average loss")
    training_plot.legend.set_title("Genome length")
    training_plot.tight_layout()
    training_plot.fig.savefig(f"experiment_results/plots/average_training_loss_run_{run_index}.pdf")

    validation_plot = sns.relplot(kind="line", data=validation_average_loss_result, x="n_epochs", y="loss", hue="genome_length").set_axis_labels("Number of epochs of training", "Average loss")
    validation_plot.legend.set_title("Genome length")
    validation_plot.tight_layout()
    validation_plot.fig.savefig(f"experiment_results/plots/average_validation_loss_run_{run_index}.pdf")

    rho_tn_vn_plot = sns.relplot(kind="line", data=rho_tn_vn_result, x="n_epochs", y="rho", hue="genome_length").set_axis_labels("Number of epochs of training", "\u03c1(tn, vn)")
    rho_tn_vn_plot.legend.set_title("Genome length")
    rho_tn_vn_plot.tight_layout()
    rho_tn_vn_plot.fig.savefig(f"experiment_results/plots/rho_tn_vn_run_{run_index}.pdf")

    rho_tn_vm_plot = sns.relplot(kind="line", data=rho_tn_vm_result, x="n_epochs", y="rho", hue="genome_length").set_axis_labels("Number of epochs of training", "\u03c1(tn, vm)")
    rho_tn_vm_plot.legend.set_title("Genome length")
    rho_tn_vm_plot.tight_layout()
    rho_tn_vm_plot.fig.savefig(f"experiment_results/plots/rho_tn_vm_run_{run_index}.pdf")

    rho_vn_vm_plot = sns.relplot(kind="line", data=rho_vn_vm_result[rho_vn_vm_result['n_epochs'] <= max_m], x="n_epochs", y="rho", hue="genome_length").set_axis_labels("Number of epochs of training", "\u03c1(vn, vm)")
    rho_vn_vm_plot.legend.set_title("Genome length")
    rho_vn_vm_plot.tight_layout()
    rho_vn_vm_plot.fig.savefig(f"experiment_results/plots/rho_vn_vm_run_{run_index}.pdf")

    rho_vn_vm_10_result = rho_vn_vm_result[rho_vn_vm_result["n_epochs"] <= 10]
    rho_vn_vm_10_plot = sns.relplot(kind="line", data=rho_vn_vm_10_result, x="n_epochs", y="rho", hue="genome_length").set_axis_labels("Number of epochs of training", "\u03c1(vn, vm)")
    rho_vn_vm_10_plot.legend.set_title("Genome length")
    rho_vn_vm_10_plot.tight_layout()
    rho_vn_vm_10_plot.fig.savefig(f"experiment_results/plots/rho_vn_vm_10_run_{run_index}.pdf")

    with open(f"experiment_results/tables/minimum_run_{run_index}.tex", "w") as fo:
        table_lines = map(
            lambda genome_length: "{genome_length} & {m} & {min_loss:.3f} \\\\".format(genome_length=genome_length, m=minima[genome_length][0], min_loss=minima[genome_length][1]),
            genome_lengths
        )
        table_lines = " \\hline\n".join(table_lines)

        table_code = f"""
            \\begin{{tabular}}{{ | c | c | c | }}
                \\hline
                Genome length & $m$ & $v_{{m}}$ \\\\
                \\hline
                \\hline
                {table_lines}
                \\hline
            \\end{{tabular}}
        """

        fo.write(table_code)

    rho_v1_vm = rho_vn_vm_result[rho_vn_vm_result["n_epochs"] == 1]
    rho_t1_vm = rho_tn_vm_result[rho_tn_vm_result["n_epochs"] == 1]

    with open(f"experiment_results/tables/rho_v1_vm_run_{run_index}.tex", "w") as fo:
        table_lines = map(
            lambda genome_length: "{genome_length} & {m} & {rho:.3f} \\\\".format(genome_length=genome_length, m=minima[genome_length][0], rho=rho_v1_vm[rho_v1_vm['genome_length'] == genome_length].at[0, 'rho']),
            genome_lengths
        )
        table_lines = " \\hline\n".join(table_lines)

        table_code = f"""
            \\begin{{tabular}}{{ | c | c | c | }}
                \\hline
                Genome length & $m$ & $\\rho(v_{{1}}, v_{{m}})$ \\\\
                \\hline
                \\hline
                {table_lines}
                \\hline
            \\end{{tabular}}
        """

        fo.write(table_code)

    with open(f"experiment_results/tables/rho_t1_vm_run_{run_index}.tex", "w") as fo:
        table_lines = map(
            lambda genome_length: "{genome_length} & {m} & {rho:.3f} \\\\".format(genome_length=genome_length, m=minima[genome_length][0], rho=rho_t1_vm[rho_t1_vm['genome_length'] == genome_length].at[0, 'rho']),
            genome_lengths
        )
        table_lines = " \\hline\n".join(table_lines)

        table_code = f"""
            \\begin{{tabular}}{{ | c | c | c | }}
                \\hline
                Genome length & $m$ & $\\rho(t_{{1}}, v_{{m}})$ \\\\
                \\hline
                \\hline
                {table_lines}
                \\hline
            \\end{{tabular}}
        """

        fo.write(table_code)
