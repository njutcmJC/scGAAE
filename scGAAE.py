# --------------------------------load the packages-----------------------------------#
import multiprocessing
import argparse
import warnings
import math
import time
import sys
import os

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.sparse as sp
import tensorflow as tf
import networkx as nx
import seaborn as sns
import anndata as ad
import pickle as pkl
import pandas as pd
import scanpy as sc
import numpy as np

from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, adjusted_rand_score, \
     normalized_mutual_info_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn import preprocessing
from tqdm import tqdm
from pylab import *

# ------------------------------------------------utils-------------------------------------------------------------------#

# ---------Data preprocessing---------#
# ---------Input:DataFrame(gene by cell) ===> Output:ndarray(cell by gene)---------#
def preprocess(data):
    var_info = pd.DataFrame(index=data.index)
    obs_info = pd.DataFrame(index=data.columns)
    adata = sc.AnnData(np.array(data.T), obs=obs_info, var=var_info)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(adata)
    return adata.X

# ---------get_mask---------#
def getmask(data):
    mask = np.ones(data.shape)
    mask[data == 0] = 0
    return mask

# ---------get_graph_data---------#
def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)
    data = adj.tocoo().data
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col

# ---------Convert Tensorflow sparse matrix to Numpy sparse matrix---------#
def conver_sparse_tf2np(input):
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]

# ---------------------------------------------------scGAAE-----------------------------------------------------------------#

class GATE():

    def __init__(self, hidden_dims, lambda_):
        self.lambda_ = lambda_
        self.n_layers = len(hidden_dims) - 1
        self.W, self.v = self.define_weights(hidden_dims)
        self.C = {}

    def __call__(self, A, X, R, S):

        # Encoder
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, H, layer)

        # Final node representations
        self.H = H

        # Decoder
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
        X_ = H

        # The reconstruction loss of node features
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))

        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        structure_loss = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))
        structure_loss = tf.reduce_sum(structure_loss)

        # Total loss
        self.loss = features_loss + self.lambda_ * structure_loss

        self.X_ = X_

        return self.loss, self.H, self.C, self.X_

    def __encoder(self, A, H, layer):
        # H = tf.nn.relu(tf.matmul(H, self.W[layer]))
        # H = tf.nn.tanh(tf.matmul(H, self.W[layer]))
        # H = tf.nn.sigmoid(tf.matmul(H, self.W[layer]))
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        # H = tf.nn.relu(tf.matmul(H, self.W[layer], transpose_b=True))
        # H = tf.nn.tanh(tf.matmul(H, self.W[layer], transpose_b=True))
        # H = tf.nn.sigmoid(tf.matmul(H, self.W[layer], transpose_b=True))
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def graph_attention_layer(self, A, M, v, layer):

        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

        # ---------------------------------------------------------------------------------------------------------------#


class Trainer():

    def __init__(self, args):

        self.args = args
        self.build_placeholders()
        gate = GATE(args.hidden_dims, args.lambda_)
        self.loss, self.H, self.C, self.X_ = gate(self.A, self.X, self.R, self.S)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, X, S, R):
        for epoch in tqdm(range(self.args.n_epochs)):
            self.run_epoch(epoch, A, X, S, R)

    def run_epoch(self, epoch, A, X, S, R):

        loss, _ = self.session.run([self.loss, self.train_op],
                                   feed_dict={self.A: A,
                                              self.X: X,
                                              self.S: S,
                                              self.R: R})

        print("Epoch: %s, Loss: %.2f" % (epoch, loss))
        return loss

    def infer(self, A, X, S, R):
        H, C, X_ = self.session.run([self.H, self.C, self.X_],
                                    feed_dict={self.A: A,
                                               self.X: X,
                                               self.S: S,
                                               self.R: R})

        return H, conver_sparse_tf2np(C), X_


# -------------------------------------------------------Input_data-------------------------------------------------------------------------#

data = pd.read_csv('/home/jc/counts.csv', index_col=0)       # dropout_data
#label = pd.read_csv('/home/jc/cell_info.csv', index_col=0)  # cell_label, if need
output_path1 = "/home/jc/imp_scGAAE_org.csv"                 # original scale
#output_path2 = "/home/jc/imp_scGAAE_model.csv"              # model output scale, if need

#celltype_label = label['Group'].values
#unique_class = np.unique(celltype_label)
#celltype_num = len(unique_class)
ngene, ncell = data.shape[0], data.shape[1]
print("The number of genes is:{}".format(ngene) + "\n" + "The number of cells is:{}".format(ncell) + "\n" + "The sparsity is:{}".format(np.mean(data.values == 0)))
#print("The number of genes is:{}".format(ngene) + "\n" + "The number of cells is:{}".format(ncell) + "\n" + 
#      "The number of cell types is:{}".format(celltype_num) + "\n" + "The sparsity is:{}".format(np.mean(data.values == 0)))

# --------------------------------------------------------Parameter settings------------------------------------------------------------------------#
n_top_genes_ = 200                  # Number of highly-variable genes to keep
n_neighbors_ = 20                   # Number of neighboring data points
n_pcs_ = 30                         # Number of principal componments

lr_ = 0.001                         # Learning rate
n_epochs_ = 200                     # Number of epochs
hidden_dims_ = [64, 32]             # Number of dimensions
lambda_ = 0

parser = argparse.ArgumentParser(description="Run gate.")
parser.add_argument('--lr', default=lr_, type=float, help='Learning rate. Default is 0.001.')
parser.add_argument('--n-epochs', default=n_epochs_, type=int, help='Number of epochs')
parser.add_argument('--hidden-dims', default=hidden_dims_, type=list, help='Number of dimensions.', nargs='+')
parser.add_argument('--lambda-', default=lambda_, type=float,
                    help='Parameter controlling the contribution of edge reconstruction in the loss function.')  
parser.add_argument('--gradient_clipping', default=5.0, type=float, help='gradient clipping')

args = parser.parse_args(args=[])
args

# ----------------------------------------------------------------------------------------------------------------------------------------------#

os.environ['KMP_WARNINGS'] = '0'

data_preprocess = preprocess(data)

mask = getmask(data)
mask_T = mask.T

var_info = pd.DataFrame(index=data.index)
obs_info = pd.DataFrame(index=data.columns)
adata = sc.AnnData(np.array(data.T), obs=obs_info, var=var_info)

#adata.obs['True_label'] = celltype_label    # add cell_label, if need

sc.pp.normalize_total(adata, target_sum=1e4)

sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_top_genes_)
adata = adata[:, adata.var.highly_variable]

sc.pp.scale(adata)

os.environ['KMP_WARNINGS'] = '0'

sc.tl.pca(adata, svd_solver='arpack')

sc.pp.neighbors(adata, n_neighbors=n_neighbors_, n_pcs=n_pcs_, metric='cosine')

# adata

# get_adjacency matrix
A = adata.obsp['connectivities']
A

# Convert get_adjacency matrix to sparse matrix
G = sp.coo_matrix(A)

# get_graph_data
G_tf, S, R = prepare_graph_data(G)

X = data_preprocess  # cell by gene

# add feature dimension size to the beginning of hidden_dims
feature_dim = X.shape[1]
args.hidden_dims = [feature_dim] + args.hidden_dims
args.hidden_dims

# Reset training
# tf.reset_default_graph()

# Train the Model
trainer = Trainer(args)
trainer(G_tf, X, S, R)

# Use the trained network for inference (reconstruction)
embeddings, attentions, X_ = trainer.infer(G_tf, X, S, R)
X.shape, X_.shape

# -------------------------------------Generate the data after imputation.------------------------------#
X_Imputed0 = mask_T * X + (1 - mask_T) * X_
print(np.min(X_Imputed0), X_Imputed0.shape)

X_Imputed1 = X_Imputed0.copy()

# Relu
X_Imputed1[X_Imputed1 < 0] = 0
print(np.min(X_Imputed1))

Gene_names = data.index
Cell_names = data.columns
# print(Gene_names,Cell_names)

imp_scGAAE = pd.DataFrame(X_Imputed1.T)
imp_scGAAE.index = Gene_names
imp_scGAAE.columns = Cell_names
# imp_scGAAE

colsum = data.apply(lambda x: x.sum())  # libsize
# colsum

imp_scGAAE_org = ((np.exp(imp_scGAAE) - 1) * colsum) / 10000
# imp_scGAAE

imp_scGAAE_org.to_csv(output_path1)     # orinal scale
#imp_scGAAE.to_csv(output_path2)        # model output scale, if need
