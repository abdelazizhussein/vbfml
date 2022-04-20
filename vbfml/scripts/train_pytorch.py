#!/usr/bin/env python3
import copy
import os
import re
import warnings
from collections import defaultdict
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from tabulate import tabulate
from torch.optim import SGD
from tqdm import tqdm
from vbfml.models_torch import Net
from vbfml.training.input import build_sequence, load_datasets_bucoffea
from vbfml.training.util import normalize_classes, save, select_and_label_datasets
import numpy as np
from sklearn import metrics

def get_training_directory(tag: str) -> str:
    return os.path.join("./output", f"model_{tag}")


def weighted_crossentropy_loss_from_labels(
    true_labels: torch.Tensor, predicted_labels: torch.Tensor, weights: torch.Tensor
):
    """
    Manual implementation of the cross entropy loss for debugging.
    """
    log1 = torch.log(predicted_labels)
    log2 = torch.log(1 - predicted_labels)

    raw_loss = -true_labels * log1 - (1 - true_labels) * log2
    raw_loss[true_labels == predicted_labels] = 0
    loss = torch.sum(weights * raw_loss) / torch.sum(weights)
    return loss


def weighted_loss(model, features, true_labels, weights, loss_criterion):
    """
    Compute the weighted loss for a batch of data.
    """
    predicted_labels = model(features)
    # print('predicted labels')
    # print(predicted_labels)
    # print('target labels')
    # print(true_labels)
   # _, targets = true_labels.max(dim=1)
    raw_loss = loss_criterion(predicted_labels, true_labels )
    loss = torch.sum(weights * raw_loss) / torch.sum(weights)
    return loss, predicted_labels


def train(
    model,
    features: torch.Tensor,
    true_labels: torch.Tensor,
    weights: torch.Tensor,
    optimizer,
    loss_criterion,
):
    """
    One iteration of training on a batch of data.
    """
    model.zero_grad()
    loss, predicted_labels = weighted_loss(
        model, features, true_labels, weights, loss_criterion
    )
    loss.backward()
    optimizer.step()
    return loss, predicted_labels


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
    #  "mjj_maxmjj",
    #  "dphijj_maxmjj",
    # "detajj_maxmjj",
    # "recoil_pt",
    # "dphi_ak40_met",
    # "dphi_ak41_met",
    # "ht",
    # "leadak4_pt",
    #  "leadak4_phi",
    # "leadak4_eta",
    # "trailak4_pt",
    #  "trailak4_phi",
    #  "trailak4_eta",
    # "leadak4_mjjmax_pt",
    #  "leadak4_mjjmax_phi",
    #  "leadak4_mjjmax_eta",
    # "trailak4_mjjmax_pt",
    #  "trailak4_mjjmax_phi",
    # "trailak4_mjjmax_eta",
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
    mjj_bins= [200, 500, 800, 1200, 1600, 2000, 2750, 3500]
    cut = 200
    slist=[]
    tlist=[]
    fp_ratios=[]
    weights=[]
    events_ratio=[]
    
    for bidx, batch in enumerate(validation_sequence):
        x, y, w = batch
        feature_values = x
        true=np.array(y)
        print(len(x))
        print(len(w))
        tlist=np.append(tlist,true[:,0])
        weights=np.append(weights,w)
        mjj_values=np.array(feature_values[:,0])
        score = (mjj_values - min(mjj_values))/max(mjj_values)
        slist=np.append(slist,mjj_values)
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
    fprw, tprw, thresholds =metrics.roc_curve(tlist,slist ,sample_weight = weights)
    tag=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    np.save('tprw'+tag,tprw)
    np.save('fprw'+tag,fprw)

    



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

simple_cut_efficency(validation_sequence)

model = Net(n_features=len(features), n_classes=2, dropout=0.2)
print(model)


# Binary cross entropy loss
# Do not apply reduction, so that we can implement
# manual weighted reduction later on
criterion =  nn.BCELoss(reduction='none')
#criterion =  nn.CrossEntropyLoss(reduction='none')
#criterion =  nn.MSELoss(reduction='none')

learning_rate = 1e-3

optm = SGD(model.parameters(), lr=learning_rate)
#optm= torch.optim.Adam(params=model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optm, factor=0.1, patience=5, threshold=1e-2, verbose=True, cooldown=2, min_lr=1e-5
)
    
# Use GPU
device = torch.device("cuda:0")
model.to(device)

history = defaultdict(list)

EPOCHS = 25

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = 0
    correct = 0

    # Put model into training mode
    model.train(True)

    # Loop over training batches and train
    
    for bidx, batch in tqdm(enumerate(training_sequence)):
        x_train, y_train, w_train = batch
        loss, predictions = train(
            model,
            torch.Tensor(x_train).to(device),
            torch.Tensor(y_train).to(device),
            torch.Tensor(w_train).to(device),
            optm,
            criterion,
        )
        epoch_loss += loss.item()

    # Put model into inference mode
    model.train(False)

    # Calculate validation loss
    validation_loss = 0
    for bidx, batch in enumerate(validation_sequence):
        x, y, w = batch
        loss, _ = weighted_loss(
            model,
            torch.Tensor(x).to(device),
            torch.Tensor(y).to(device),
            torch.Tensor(w).to(device),
            criterion,
        )
        validation_loss += loss.item()

    # Pass validation loss to learning rate scheduler
    scheduler.step(validation_loss)

    print(f"Epoch {epoch+1} Loss : {epoch_loss} Validation loss : {validation_loss}")

    history["x_loss"].append(epoch)
    history["x_val_loss"].append(epoch)
    history["y_loss"].append(loss)
    history["y_val_loss"].append(validation_loss)


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
torch.save(model, prepend_path("model.pt"))
