import os
from dataclasses import dataclass

from numpy.core.fromnumeric import ndim

import mplhep as hep
import numpy as np
from hist import Hist
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import integrate
from sklearn import metrics
import time
plt.style.use(hep.style.CMS)


def keys_to_list(keys):
    return list(sorted(map(str, keys)))


def merge_flow_bins(values):
    new_values = values[1:-1]
    new_values[0] = new_values[0] + values[0]
    new_values[-1] = new_values[-1] + values[-1]
    return new_values


colors = ["b", "r", "k", "g"]


@dataclass
class TrainingHistogramPlotter:
    """
    Tool to make standard plots based on histograms created with
    the TrainingAnalyzer
    """
    weights:"list"
    predicted_scores:"list"
    validation_scores:"list"
    histograms: "list[Hist]"
    output_directory: str = "./output"

    def __post_init__(self):
        self.sequence_types = keys_to_list(self.histograms.keys())
        histogram_names = []
        for sequence_type in self.sequence_types:
            histogram_names.extend(keys_to_list(self.histograms[sequence_type].keys()))
        self.histogram_names = sorted(list(set(histogram_names)))

        self.output_directory = os.path.abspath(self.output_directory)

    def create_output_directory(self, subdirectory: str) -> None:
        directory = os.path.join(self.output_directory, subdirectory)
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

    def get_histogram(self, sequence_type: str, name: str) -> Hist:
        return self.histograms[sequence_type][name]

    def get_histograms_across_sequences(self, histogram_name: str) -> "list[Hist]":
        histograms = {}
        for sequence_type in self.sequence_types:
            histograms[sequence_type] = self.get_histogram(
                sequence_type, histogram_name
            )
        return histograms


    def plot_by_sequence_types(self) -> None:
        """
        Comparison plots between the training and validation sequences.
        Generates plots for features + tagger scores.
        """
        output_subdir = "by_sequence"
        self.create_output_directory(output_subdir)

        for name in tqdm(self.histogram_names, desc="Plotting histograms"):
            fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=(10, 10),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True,
            )
            histograms = self.get_histograms_across_sequences(name)

            for sequence, histogram in histograms.items():
                label_axis = [axis for axis in histogram.axes if axis.name == "label"][
                    0
                ]

                for label in label_axis.edges[:-1]:
                    label = int(label)
                    color = colors[label % len(colors)]

                    histogram_for_label = histogram[{"label": int(label)}]

                    edges = histogram_for_label.axes[0].edges
                    values = merge_flow_bins(histogram_for_label.values(flow=True))

                    if np.all(values == 0):
                        continue
                    variances = merge_flow_bins(
                        histogram_for_label.variances(flow=True)
                    )

                    norm = np.sum(values)

                    if sequence == "validation":
                        hep.histplot(
                            values / norm,
                            edges,
                            yerr=np.sqrt(variances) / norm,
                            histtype="errorbar",
                            label=f"class={label}, {sequence}",
                            marker="o",
                            color=color,
                            ax=ax,
                            markersize=5,
                        )
                    else:
                        values_normalized = values/norm
                        histogram1 = hep.histplot(values/norm, edges, color=color, label=f"class={label}, {sequence}",ls="-", ax=ax)       
                        steppatch = (histogram1[0][0]).get_data()
                        values_hist = np.reshape(np.asarray(steppatch[0]),(len(steppatch[0]),1))
                        edges_hist= np.reshape((np.asarray(steppatch[1])),(len(steppatch[1]),1))
                        integral = np.zeros((len(edges_hist),1))
                        #compute efficency
                        for bin in range(1,len(values_hist)+1):
                            integral[bin-1] = (sum(values_hist[0:bin] * np.diff(edges_hist[0:bin+1],axis=0)))                               
                        rax.plot(edges_hist[1:],integral[:-1],color = color, label=f"class={label}, {sequence}",axes=rax)
                    ax.legend()
                    rax.legend(prop={'size': 15})
                    rax.set_xlabel(name)
                    rax.set_ylabel("Efficency (a.u.)")
                    ax.set_ylabel("Events (a.u.)")
                    ax.set_yscale("log")
                    for ext in "pdf", "png":
                        fig.savefig(
                            os.path.join(self.output_directory, output_subdir, f"{name}.{ext}")
                        )
                    plt.close(fig)

    def plot(self) -> None:
        self.plot_by_sequence_types()
        self.plot_ROC()

    #plot ROC curve for each class
    def plot_ROC(self)->None:
        fpr=dict()
        tpr=dict()
        score_classes = np.array(self.predicted_scores).shape[2]
        number_batches=np.array(self.predicted_scores).shape[0]

        for sclass in range(score_classes):
            #combine batch scores for each class
            p_score =np.array(self.predicted_scores[0][:,sclass])
            v_score = np.array(self.validation_scores[0][:,sclass])
            sample_weights = np.array(self.weights)
            predicted_scores_combined =  p_score
            validation_scores_combined =  v_score
            for batch in range(1, number_batches):
                sample_weights=np.append(sample_weights,sample_weights)
                p_score =np.asarray(self.predicted_scores[batch][:,sclass])
                v_score = np.asarray(self.validation_scores[batch][:,sclass])  
                predicted_scores_combined = np.append(predicted_scores_combined, p_score)
                validation_scores_combined = np.append(validation_scores_combined,v_score)
            np.reshape(predicted_scores_combined,(len(predicted_scores_combined),1))
            np.reshape(validation_scores_combined,(len(validation_scores_combined),1))
            #make ROC curve         
            fpr[sclass],tpr[sclass],thresholds=metrics.roc_curve(validation_scores_combined,predicted_scores_combined,sample_weight=sample_weights)
            auc = metrics.roc_auc_score(validation_scores_combined,predicted_scores_combined,sample_weight=sample_weights)
            fig, ax = plt.subplots()
            label_1 = "Current Classifier- AUC ="+str(round(auc,3))
            ax.plot(fpr[sclass],tpr[sclass],label=label_1)
            ax.set_title("Label "+str(sclass))
            ax.set_ylabel('True Positive Rate')
            ax.set_xlabel('False Positive Rate')
            x =np.linspace(0, 1, 20)
            ax.plot(x,x,linestyle='--',label='Random Classifier')
            ax.legend()
            outdir = os.path.join(self.output_directory, "ROC")
            try:
                os.makedirs(outdir)
            except FileExistsError:
                pass

            for ext in "png", "pdf":
                fig.savefig(os.path.join(outdir, f"ROC_score_{sclass}.{ext}"))

def plot_history(history, outdir):

    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()

    x = np.array(range(len(history["loss"])))

    ax.plot(x, history["loss"], color="b", ls="--", label="Training")
    ax.plot(x, history["val_loss"], color="b", label="Validation", marker="o")

    ax2 = ax.twinx()
    ax2.plot(x, history["categorical_accuracy"], color="r", ls="--", label="Training")
    ax2.plot(x, history["val_categorical_accuracy"], color="r", label="Validation", marker="o")
    ax.set_xlabel("Training time (a.u.)")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax.legend(title="Loss")
    ax2.legend(title="Accuracy")
    outdir = os.path.join(outdir, "history")
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    for ext in "png", "pdf":
        fig.savefig(os.path.join(outdir, f"history.{ext}"))