from typing import Dict, Tuple, List
import os
from dataclasses import dataclass
import matplotlib
import mplhep as hep
import numpy as np
from hist import Hist
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import integrate
from sklearn import metrics
import time
from scipy.interpolate import interp1d

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

    weights: "list"
    predicted_scores: "list"
    validation_scores: "list"
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
                        values_normalized = values / norm
                        histogram1 = hep.histplot(
                            values / norm,
                            edges,
                            color=color,
                            label=f"class={label}, {sequence}",
                            ls="-",
                            ax=ax,
                        )
                        steppatch = (histogram1[0][0]).get_data()
                        values_hist = np.reshape(
                            np.asarray(steppatch[0]), (len(steppatch[0]), 1)
                        )
                        edges_hist = np.reshape(
                            (np.asarray(steppatch[1])), (len(steppatch[1]), 1)
                        )
                        integral = np.zeros((len(edges_hist), 1))
                        # compute efficency
                        for bin in range(1, len(values_hist) + 1):
                            integral[bin - 1] = sum(
                                values_hist[0:bin]
                                * np.diff(edges_hist[0 : bin + 1], axis=0)
                            )
                        rax.plot(
                            edges_hist[1:],
                            integral[:-1],
                            color=color,
                            label=f"class={label}, {sequence}",
                            axes=rax,
                        )
                    ax.legend()
                    rax.legend(prop={"size": 15})
                    rax.set_xlabel(name)
                    rax.set_ylabel("Efficency (a.u.)")
                    ax.set_ylabel("Events (a.u.)")
                    ax.set_yscale("log")
                    #ax.axis(ymin=10e-5,ymax=10e-1)
                    for ext in "pdf", "png":
                        fig.savefig(
                            os.path.join(
                                self.output_directory, output_subdir, f"{name}.{ext}"
                            )
                        )
                    plt.close(fig)

    def _covariance_dict_to_matrices(
        self, covariance: Dict[Tuple[str, str], float]
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        feature_names = sorted(set([x[0] for x in covariance.keys()]))
        n_feat = len(feature_names)

        matrices = {}
        matrices["covariance"] = np.zeros((n_feat, n_feat))
        matrices["correlation"] = np.zeros((n_feat, n_feat))

        for ifeat1, feat1 in enumerate(feature_names):
            for ifeat2, feat2 in enumerate(feature_names):
                key = (feat1, feat2)
                if key not in covariance:
                    key = (feat2, feat1)
                cov = covariance[key][0, 1]
                v1 = covariance[(feat1, feat1)][0, 1]
                v2 = covariance[(feat2, feat2)][0, 1]

                matrices["covariance"][ifeat1, ifeat2] = cov
                matrices["correlation"][ifeat1, ifeat2] = cov / np.sqrt(v1 * v2)

        return feature_names, matrices

    def plot_covariance(
        self, covariance: Dict[int, Dict[Tuple[str, str], float]]
    ) -> None:

        ncolors = 10
        cmap = plt.get_cmap("RdBu", ncolors)
        for class_index, class_covariance in covariance.items():
            feature_names, matrices = self._covariance_dict_to_matrices(
                class_covariance
            )

            output_subdirectory = os.path.join(self.output_directory, "covariance")
            try:
                os.makedirs(output_subdirectory)
            except FileExistsError:
                pass

            for tag, matrix in matrices.items():
                fig = plt.figure()
                ax = plt.gca()

                if "correlation" in tag:
                    vmin, vmax = -1, 1
                else:
                    vmax = 1.2 * np.max(np.abs(matrix))
                    vmin = -vmax

                cax = ax.matshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
                cbar = fig.colorbar(cax, shrink=0.6)
                cbar.set_label(tag)
                cbar.set_ticks(
                    [vmin + i * (vmax - vmin) / ncolors for i in range(ncolors + 1)]
                )
                plt.xticks(
                    list(range(len(feature_names))), feature_names, rotation="vertical"
                )
                plt.yticks(
                    list(range(len(feature_names))),
                    feature_names,
                    rotation="horizontal",
                )
                plt.subplots_adjust(left=0.3, bottom=0.0, right=0.9, top=0.8)

                for i in range(matrix.shape[0]):
                    ax.axhline(i + 0.5, color="k")
                    ax.axvline(i + 0.5, color="k")
                for ext in "pdf", "png":
                    fig.savefig(
                        os.path.join(
                            output_subdirectory,
                            f"feature_{tag}_class{class_index}.{ext}",
                        )
                    )

                plt.close(fig)

    def plot(self) -> None:
        self.plot_by_sequence_types()
        self.plot_ROC()

    # plot ROC curve for each class
    def plot_ROC(self) -> None:
        fpr = dict()
        tpr = dict()
        score_classes = np.array(self.predicted_scores).shape[2]
        number_batches = np.array(self.predicted_scores).shape[0]
        for sclass in tqdm(range(score_classes), desc="Plotting ROC curves"):
            # combine batch scores for each class
            p_score = np.array(self.predicted_scores[0])
            v_score = np.array(self.validation_scores[0])
            sample_weights = np.array(self.weights)
            predicted_scores_combined = p_score
            validation_scores_combined = v_score
            for batch in range(1, number_batches):
                sample_weights = np.append(sample_weights, sample_weights)
                p_score = np.asarray(self.predicted_scores[batch])
                v_score = np.asarray(self.validation_scores[batch])
                predicted_scores_combined = np.append(
                    predicted_scores_combined, p_score
                )
                validation_scores_combined = np.append(
                    validation_scores_combined, v_score
                )
           # np.reshape(predicted_scores_combined, (len(predicted_scores_combined), 1))
            #np.reshape(validation_scores_combined, (len(validation_scores_combined), 1))
            # make ROC curve
            fpr[sclass], tpr[sclass], thresholds = metrics.roc_curve(
                validation_scores_combined[:,0],
                predicted_scores_combined[:,0],
                sample_weight=sample_weights,
            )
            auc = metrics.roc_auc_score(
                validation_scores_combined[:,0],
                predicted_scores_combined[:,0],
                sample_weight=sample_weights,
            )
            tpcut1 = np.load('tprw2022-02-22_23-57-01.npy')
            fpcut1 = np.load('fprw2022-02-22_23-57-01.npy')
            ay= plt.figure()
            x = np.linspace(0.0112,0.2,100)
            f={}
            f["mjj"] = interp1d(fpcut1,tpcut1)
            f["xgb"] = interp1d(fpr[0], tpr[0])
            plt.plot(x,f["xgb"](x)/f["mjj"](x))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate NN / True Positive Rate Mjj cut')

            #plt.xlim(0,0.10)
            fig, ax = plt.subplots()
            label_1 = "Current Classifier- AUC=" + str(round(auc, 3))
            ax.plot(fpr[sclass],tpr[sclass], label=label_1)
            ax.set_title("Label " + str(sclass))
            ax.set_ylabel("True Positive Rate")
            ax.set_xlabel("False Positive Rate")
            x = np.linspace(0, 1, 20)
            ax.plot(x, x, linestyle="--", label="Random Classifier")
            
            #ax.plot(fpcut,tpcut,color='r',linestyle='dashdot',label='mjj cut')
            ax.axis(xmin=0,xmax=0.2)
            ax.plot(fpcut1,tpcut1,color='r',linestyle='dashdot',label='mjj cut1')
            #ax.set_xlim(0,0.2)
            ax.legend()
            
            outdir = os.path.join(self.output_directory, "ROC")
            try:
                os.makedirs(outdir)
            except FileExistsError:
                pass

            for ext in "png", "pdf":
                fig.savefig(os.path.join(outdir, f"ROC_score_{sclass}.{ext}"))
                ay.savefig(os.path.join(outdir, f"ROC_score1_{sclass}.{ext}"))


def plot_history(history, outdir):

    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()
    ax.plot(
        history["x_loss"],
        history["y_loss"],
        color="b",
        ls="--",
        marker="s",
        label="Training",
    )
    ax.plot(
        history["x_val_loss"],
        history["y_val_loss"],
        color="b",
        label="Validation",
        marker="o",
    )

    ax.set_xlabel("Training time (a.u.)")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(title="Loss")
    outdir = os.path.join(outdir, "history")
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    for ext in "png", "pdf":
        fig.savefig(os.path.join(outdir, f"history_loss.{ext}"))