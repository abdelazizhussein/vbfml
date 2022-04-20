#!/usr/bin/env python3
import copy
import os
import re
import warnings
from collections import defaultdict
from datetime import datetime
import pandas as pd
#import torch
#import torch.nn as nn
from tabulate import tabulate
from torch.optim import SGD
from tqdm import tqdm
from vbfml.models_torch import Net
from vbfml.training.input import build_sequence, load_datasets_bucoffea
from vbfml.training.util import normalize_classes, save, select_and_label_datasets
import numpy as np
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import mplhep as hep
import hist

def get_training_directory(tag: str) -> str:
    return os.path.join("./output", f"model_{tag}")


# def weighted_crossentropy_loss_from_labels(
#     true_labels: torch.Tensor, predicted_labels: torch.Tensor, weights: torch.Tensor
# ):
#     """
#     Manual implementation of the cross entropy loss for debugging.
#     """
#     log1 = torch.log(predicted_labels)
#     log2 = torch.log(1 - predicted_labels)

#     raw_loss = -true_labels * log1 - (1 - true_labels) * log2
#     raw_loss[true_labels == predicted_labels] = 0
#     loss = torch.sum(weights * raw_loss) / torch.sum(weights)
#     return loss


# def weighted_loss(model, features, true_labels, weights, loss_criterion):
#     """
#     Compute the weighted loss for a batch of data.
#     """
#     predicted_labels = model(features)
#     # print('predicted labels')
#     # print(predicted_labels)
#     # print('target labels')
#     # print(true_labels)
#    # _, targets = true_labels.max(dim=1)
#     raw_loss = loss_criterion(predicted_labels, true_labels )
#     loss = torch.sum(weights * raw_loss) / torch.sum(weights)
#     return loss, predicted_labels


# def train(
#     model,
#     features,
#     true_labels,
#     weights,
#     optimizer,
#     loss_criterion,
# ):
#     """
#     One iteration of training on a batch of data.
#     """
#     model.zero_grad()
#     loss, predicted_labels = weighted_loss(
#         model, features, true_labels, weights, loss_criterion
#     )
#     loss.backward()
#     optimizer.step()
#     return loss, predicted_labels


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

all_datasets = load_datasets_bucoffea("./data2")

dataset_labels = {
    "ewk_17": "(EWK.*2017|VBF_HToInvisible_M125_withDipoleRecoil_pow_pythia8_2017)",
    "v_qcd_nlo_17": "(WJetsToLNu_Pt-\d+To.*|Z\dJetsToNuNu_M-50_LHEFilterPtZ-\d+To\d+)_MatchEWPDG20-amcatnloFXFX_2017",
}

# dataset_labels = {
#     "ewk_17": "EWK.*2017",
#     "v_qcd_nlo_17": "(WJetsToLNu_Pt-\d+To.*|Z\dJetsToNuNu_M-50_LHEFilterPtZ-\d+To\d+)_MatchEWPDG20-amcatnloFXFX_2017",
#     "signal_17": "VBF_HToInvisible_M125_withDipoleRecoil_pow_pythia8_2017",
# }
datasets = select_and_label_datasets(all_datasets, dataset_labels)
# for dataset_info in datasets:
#     if re.match(dataset_labels["v_qcd_nlo_17"], dataset_info.name):
#         dataset_info.n_events = 0.02 * dataset_info.n_events
features = [
    "mjj",
    "dphijj",
    "detajj",
     "mjj_maxmjj",
     "dphijj_maxmjj",
    "detajj_maxmjj",
    "recoil_pt",
    "dphi_ak40_met",
    "dphi_ak41_met",
    "ht",
    "leadak4_pt",
     "leadak4_phi",
    "leadak4_eta",
    "trailak4_pt",
     "trailak4_phi",
     "trailak4_eta",
    "leadak4_mjjmax_pt",
     "leadak4_mjjmax_phi",
     "leadak4_mjjmax_eta",
    "trailak4_mjjmax_pt",
     "trailak4_mjjmax_phi",
    "trailak4_mjjmax_eta",
]

training_sequence = build_sequence(
    datasets=copy.deepcopy(datasets),
    features=features,
    weight_expression="weight_total*xs/sumw",
)
validation_sequence = build_sequence(
    datasets=copy.deepcopy(datasets),
    features=features,
    weight_expression="weight_total*xs/sumw",
)
validation1_sequence = build_sequence(
    datasets=copy.deepcopy(datasets),
    features=features,
    weight_expression="weight_total*xs/sumw",
)


#mjj cut comparison
def simple_cut_efficency(validation_sequence):
    # mjj_bins= [200, 500, 800, 1200, 1600, 2000, 2750, 3500]
    # cut = 200
    # slist=[]
    # tlist=[]
    # fp_ratios=[]
    # weights=[]
    # events_ratio=[]
    
    # for bidx, batch in enumerate(validation_sequence):
    #     x, y, w = batch
    #     feature_values = x
    #     true=np.array(y[:,0])
    #     tlist=np.append(tlist,true)
    #     weights=np.append(weights,w)
    #     mjj_values=np.array(feature_values[:,0])
    #     score = (mjj_values - min(mjj_values))/max(mjj_values)
    #     slist=np.append(slist,score)
    #         nsignal_events = np.sum(mjj_values>cut)
    #         nclassified_signal += nsignal_events
    #         classified_signal_events_true_label = np.array(true_labels[np.where(mjj_values>cut)])
    #         classified_notsignal_events_true_label = np.array(true_labels[np.where(mjj_values<cut)])
    #         false_neg_signal_count = np.sum(classified_notsignal_events_true_label[:,0]>0)
    #         nfn_singal+=false_neg_signal_count
    #         true_neg_signal_count= np.sum(classified_notsignal_events_true_label[:,0]<1)
    #         ntn_signal+=true_neg_signal_count
    #         true_signal_count=np.sum(classified_signal_events_true_label[:,0]>0)
    #         ntrue_signal += true_signal_count
    #         fp_signal_count=np.sum(classified_signal_events_true_label[:,0]<1)
    #         nfp_signal+=fp_signal_count
    #     tp_ratios.append(ntrue_signal/(ntrue_signal+nfn_singal))
    #     fp_ratios.append(nfp_signal/(ntn_signal+nfp_signal)) 
    #     events_ratio.append(ntrue_signal/np.sum(true_labels[:,0]>0))   
    #     cut+=1
    y = validation_sequence[0][1]
    features=validation_sequence[0][0]
    truelabel = y.argmin(axis=1)
    mjj_values=np.array(features[:,0])
    weights = validation_sequence[0][2]
    score = (mjj_values - min(mjj_values))/max(mjj_values)
    fprw, tprw, thresholds = metrics.roc_curve(truelabel, score,sample_weight=weights)
    #fprw, tprw, thresholds = metrics.roc_curve(tlist, slist,sample_weight = weights)
    return fprw,tprw
    tag=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # np.save('tpr'+tag,tpr)
    # np.save('fpr'+tag,fpr)
    # np.save('tprw'+tag,tprw)
    # np.save('fprw'+tag,fprw)

    



validation1_sequence.read_range = (0.8, 1.0)




normalize_classes(training_sequence, target_integral=1e6)
normalize_classes(validation_sequence, target_integral=1e6)

# Training sequence
training_sequence.read_range = (0.0, 0.8)
training_sequence.scale_features = True
training_sequence.batch_size = 25
training_sequence.batch_buffer_size = 5e5
training_sequence[0]

# Validation sequence
validation_sequence.read_range = (0.8, 1.0)
validation_sequence.scale_features = True
validation_sequence._feature_scaler = copy.deepcopy(training_sequence._feature_scaler)
validation_sequence.batch_size = 1e4
validation_sequence.batch_buffer_size = 100


#model = Net(n_features=len(features), n_classes=2, dropout=0.2)
training_sequence.batch_size=int(1e8)
validation_sequence.batch_size=int(1e8)



# Binary cross entropy loss
# Do not apply reduction, so that we can implement
# manual weighted reduction later on
#criterion =  nn.BCELoss(reduction='none')
#criterion =  nn.CrossEntropyLoss(reduction='none')
#criterion =  nn.MSELoss(reduction='none')

learning_rate = 1e-3

# optm = SGD(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optm, factor=0.1, patience=5, threshold=1e-2, verbose=True, cooldown=2, min_lr=1e-5
# )
    
# Use GPU
# device = torch.device("cuda:0")
#model.to(device)

history = defaultdict(list)

EPOCHS = 50
# Training loop
model = xgb.XGBRegressor(use_label_encoder=False,learning_rate=1e-3, n_estimators=1000,max_depth=10,objective="binary:logitraw")
x,y,w=training_sequence[0]
print("hey!")
print(len(x))

xtest,ytest,wtest = validation_sequence[0]
print(ytest[0])
print(ytest[1])
print(ytest[2])
print(ytest[3])
print(ytest[0:4].argmin(axis=1))
model.fit(x,y.argmax(axis=1), sample_weight=w, eval_set=[(xtest,ytest.argmax(axis=1))],sample_weight_eval_set=[wtest],early_stopping_rounds=50)

all_datasets = load_datasets_bucoffea("./real_data")
training_sequence = build_sequence(
    datasets=copy.deepcopy(all_datasets),
    features=features,
    weight_expression="weight_total*xs/sumw",
)

training_sequence.read_range = (0.0, 1)
training_sequence.scale_features = True
training_sequence.batch_size = 25
training_sequence.batch_buffer_size = 5e5
training_sequence[0]

histogram = hist.Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="score"))
for bidx, batch in tqdm(enumerate(training_sequence)):
    x_train, y_train, w_train = batch
    scores = model.predict(x_train) 
    histogram.fill(score=scores)

def merge_flow_bins(values):
    new_values = values[1:-1]
    new_values[0] = new_values[0] + values[0]
    new_values[-1] = new_values[-1] + values[-1]
    return new_values
variances = merge_flow_bins(
                        histogram.variances(flow=True)
                    )
edges = histogram.axes[0].edges
values = merge_flow_bins(histogram.values(flow=True))   
norm = np.sum(values)

fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=(10, 10),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True,
)
hep.histplot(
                values / norm,
                edges,
                yerr=np.sqrt(variances) / norm,
                histtype="errorbar",
                label="score",
                marker="o",
                ax=ax,
                markersize=5,
                        )

ax.legend()
ax.set_ylabel("Events (a.u.)")
#ax.axis(ymin=10e-5,ymax=10e-1)A
for ext in "pdf", "png":
    fig.savefig(
        os.path.join(
             "./output/model_2022-04-02_23-15/plots", "by_sequence","real_scores_bdt.png"
        )
    )
plt.close(fig)

pred_test = model.predict(xtest)
ytrue = ytest.argmin(axis=1)
series_pred_test = pd.Series(pred_test)
bins = np.linspace(0,1,25)
fpr1={}
tpr1={}
for ifeat, feature in enumerate(features):
     fpr, tpr, thresholds = metrics.roc_curve(y_true=ytest.argmin(axis=1), y_score=xtest[:,ifeat], sample_weight=wtest)
     tpr1[feature] = tpr
     fpr1[feature] = fpr
#fp,tp = simple_cut_efficency(validation_sequence)

fprw, tprw, thresholds = metrics.roc_curve(ytest.argmin(axis=1),pred_test ,sample_weight = wtest)
fpr, tpr, thresholds = metrics.roc_curve(ytest.argmin(axis=1),xtest[:,0] ,sample_weight = wtest)
auc = metrics.roc_auc_score(
                ytest.argmin(axis=1),pred_test,
                sample_weight=wtest,
            )
label_1 = "BDT- AUC =" + str(round(auc, 3))
plt.plot(1-fprw,1-tprw,color='r',linestyle='dashdot',label=label_1)
x = np.linspace(0, 1, 20)
plt.plot(x, x, linestyle="--", label="Random Classifier")
plt.plot(fpr1["mjj"],tpr1["mjj"], linestyle="--", label="mjj")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend()
plt.xlim(0,0.5)
plt.savefig("plot1_bdt")

plt.figure(2)
x = np.linspace(0.05,0.2,100)
f={}
f["mjj"] = interp1d(fpr1["mjj"],tpr1["mjj"])
f["xgb"] = interp1d(1-fprw, 1-tprw)
plt.plot(x,f["xgb"](x)/f["mjj"](x))
plt.ylabel("Classifier Positive Rate / mjj Positive Rate ")
plt.savefig("plot2_bdt")
ytrue= ytest.argmax(axis=1)
for iclass in 0,1:
    mask = ytrue==iclass
    series_pred_test[mask].plot(kind="hist",alpha=0.5,bins=bins,weights=wtest[mask],label="Class"+str(iclass))
    plt.xlabel("score_0_bdt")
    plt.legend()
    plt.savefig("plot_bdt")
    



# predictions = [round(value) for value in y_pred]
# evaluate predictions
#accuracy = metrics.accuracy_score(ytest.argmin(axis=1), predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # Put model into inference mode
    # model.train(False)

    # Calculate validation loss
    # validation_loss = 0
    # for bidx, batch in enumerate(validation_sequence):
    #     x, y, w = batch
    #     loss, _ = weighted_loss(
    #         model,
    #         torch.Tensor(x).to(device),
    #         torch.Tensor(y).to(device),
    #         torch.Tensor(w).to(device),
    #         criterion,
    #     )
       # validation_loss += loss.item()

    # Pass validation loss to learning rate scheduler

    #print(f"Epoch {epoch+1} Loss : {epoch_loss} Validation loss : {validation_loss}")

    #history["x_loss"].append(epoch)
    #history["x_val_loss"].append(epoch)
    # history["y_loss"].append(loss)
    # history["y_val_loss"].append(validation_loss)

print("training done!")

# Save everything
tag=datetime.now().strftime("%Y-%m-%d_%H-%M")
training_directory = get_training_directory(tag)


def prepend_path(fname):
    return os.path.join(training_directory, fname)


try:
    os.makedirs(training_directory)
except FileExistsError:
    pass

# Feature scaling object for future evaluation
save(training_sequence._feature_scaler, prepend_path("feature_scaler.pkl"))

# List of features
save(
    features,
    prepend_path(
        "features.pkl",
    ),
)

# Training and validation sequences
# Clear buffer before saving to save space
for seq in training_sequence, validation_sequence:
    seq.buffer.clear()
save(training_sequence, prepend_path("training_sequence.pkl"))
save(validation_sequence, prepend_path("validation_sequence.pkl"))
save(history, prepend_path("history.pkl"))
#torch.save(model, prepend_path("model.pt"))
