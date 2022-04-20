import sys
import os
import tensorflow as tf
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
from vbfml.training.data import TrainingLoader
import numpy as np
loader = TrainingLoader(sys.argv[1])
history = loader.get_history()
sequence_types = "training", "validation"


model = loader.get_model()
table = []
val_loss=[]
loss_a=[]
def l1l2_weight_loss(model):
    l1l2_loss = 0
    for layer in model.layers:
        if 'layer' in layer.__dict__ or 'cell' in layer.__dict__:
            l1l2_loss += _l1l2_rnn_loss(layer)
            continue
            
        if 'kernel_regularizer' in layer.__dict__ or \
           'bias_regularizer'   in layer.__dict__:
            l1l2_lambda_k, l1l2_lambda_b = [0,0], [0,0] # defaults
            if layer.__dict__['kernel_regularizer'] is not None:
                l1l2_lambda_k = list(layer.kernel_regularizer.__dict__.values())
            if layer.__dict__['bias_regularizer']   is not None:
                l1l2_lambda_b = list(layer.bias_regularizer.__dict__.values())
                
            if any([(_lambda != 0) for _lambda in (l1l2_lambda_k + l1l2_lambda_b)]):
                W = layer.get_weights()
    
                for idx,_lambda in enumerate(l1l2_lambda_k + l1l2_lambda_b):
                    if _lambda != 0:
                        _pow = 2**(idx % 2) # 1 if idx is even (l1), 2 if odd (l2)
                        l1l2_loss += _lambda*np.sum(np.abs(W[idx//2])**_pow)
    return l1l2_loss
def _l1l2_rnn_loss(layer):
    l1l2_loss = 0
    if 'backward_layer' in layer.__dict__:
        bidirectional = True
        _layer = layer.layer
    else:
        _layer = layer
        bidirectional = False
    ldict = _layer.cell.__dict__
        
    if 'kernel_regularizer'    in ldict or \
       'recurrent_regularizer' in ldict or \
       'bias_regularizer'      in ldict:
        l1l2_lambda_k, l1l2_lambda_r, l1l2_lambda_b = [0,0], [0,0], [0,0]
        if ldict['kernel_regularizer']    is not None:
            l1l2_lambda_k = list(_layer.kernel_regularizer.__dict__.values())
        if ldict['recurrent_regularizer'] is not None:
            l1l2_lambda_r = list(_layer.recurrent_regularizer.__dict__.values())
        if ldict['bias_regularizer']      is not None:
            l1l2_lambda_b = list(_layer.bias_regularizer.__dict__.values())
        
        all_lambda = l1l2_lambda_k + l1l2_lambda_r + l1l2_lambda_b
        if any([(_lambda != 0) for _lambda in all_lambda]):
            W = layer.get_weights()
            idx_incr = len(W)//2 # accounts for 'use_bias'
            
            for idx,_lambda in enumerate(all_lambda):
                if _lambda != 0:
                    _pow = 2**(idx % 2) # 1 if idx is even (l1), 2 if odd (l2)
                    l1l2_loss += _lambda*np.sum(np.abs(W[idx//2])**_pow)
                    if bidirectional:
                        l1l2_loss += _lambda*np.sum(
                                    np.abs(W[idx//2 + idx_incr])**_pow)
    return l1l2_loss 
print(model.losses)
for sequence_type in sequence_types:
    cce = tf.keras.losses.CategoricalCrossentropy() 
    #ca = tf.losses.CategoricalAccuracy()

    cce_now = tf.keras.losses.CategoricalCrossentropy()
   # ca_now = tf.keras.losses.CategoricalAccuracy()
    sequence = loader.get_sequence(sequence_type)
    sequence.batch_size = 1e6
    sequence.batch_buffer_size = 20

    loss = 0
    print(len(sequence))
    for ibatch in tqdm(range(len(sequence))):
        x, ytrue, w = sequence[ibatch]
        ypred = model.predict(x)
        losss = cce(ytrue, ypred, sample_weight=w)
        #ca.update_state(ytrue, ypred, sample_weight=w)
        lossss = cce_now(ytrue, ypred,sample_weight=w)
       # ca_now.update_state(ytrue, ypred)
    


    # fig, ax = plt.subplots()
    # ax.plot(np.array(val_loss),range(len(sequence)),label="val_loss")
    # ax.plot(np.array(loss_a),range(len(sequence)),label="training_Loss")
    # ax.legend()
    # outdir = "./manual_loss"
    # try:
    #     os.makedirs(outdir)
    # except FileExistsError:
    #     pass

    # for ext in "png", "pdf":
    #     fig.savefig(os.path.join(outdir, f"loss{sequence_type}.{ext}"))

    tag = "val_" if sequence_type == "validation" else ""

    line = [
        sequence_type,
        history[f"y_{tag}loss"][-1],
        losss,
        lossss,
        history[f"y_{tag}categorical_crossentropy"][-1],
    ]
    table.append(line)
headers = [
    "Sequence",
    "Loss from history",
    "Manual loss (weighted)",
    "Manual loss (unweighted)",
    "Metric loss from history",
]
print(tabulate(table, headers=headers))
