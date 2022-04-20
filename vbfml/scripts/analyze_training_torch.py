#!/usr/bin/env python3

import os

import click

from vbfml.training.analysis_torch import TrainingAnalyzer
from vbfml.training.data_torch import TrainingLoader
from vbfml.training.plot_torch import TrainingHistogramPlotter, plot_history


@click.group()
def cli():
    pass


@cli.command()
@click.argument("training_path")
def analyze(training_path):
    analyzer = TrainingAnalyzer(training_path)
    analyzer.analyze()
    analyzer.write_to_cache()


@cli.command()
@click.argument("training_path")
@click.option("--force-analyze", default=False, is_flag=True)
def plot(training_path: str, force_analyze: bool = False):
    # Redo the analysis if cache does not exist
    analyzer = TrainingAnalyzer(training_path)
    if force_analyze or not analyzer.load_from_cache():
        analyzer.analyze()
        analyzer.write_to_cache()

    # # Plot histograms
    output_directory = os.path.join(training_path, "plots")
    plotter = TrainingHistogramPlotter(
        analyzer.data["weights"],
        analyzer.data["predicted_scores"],
        analyzer.data["validation_scores"],
        analyzer.data["histograms"],
        output_directory,
    )
    plotter.plot()
    plotter.plot_covariance(analyzer.data["covariance"])

    # Plot training history
    loader = TrainingLoader(training_path)
    plot_history(loader.get_history(), output_directory)


if __name__ == "__main__":
    cli()