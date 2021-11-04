"""Program for conducting Link Prediction over datasets downloaded from AsymProj

It constructs implicit matrix M^{WYS}, as described in our paper.

These datasets originally come from node2vec but we use the AsymProj train/test
partitions for consistency (node2vec, unfortunately, did not publish the splits
however they exactly decribed the partitioning procedure which was replicated
by AsymProj).

The program reads the dataset, trains the model (using functional SVD),
fine-tunes the network over singular values,
computes the AUC-ROC on test edges, and prints the AUC-ROC metric as well as
training time (i.e. spent inside the functional SVD computation).
"""

import json
import os
import sys
import time

from absl import app, flags
import numpy as np
import scipy.sparse
import tensorflow as tf

import tf_fsvd


if __name__ == '__main__':
  flags.DEFINE_string('dataset_name', 'ppi', 'Must be directory inside datasets_dir')
  flags.DEFINE_string(
      'datasets_dir', 'datasets/asymproj', 'Directory containing AsymProj datasets.')
  flags.DEFINE_integer('window', 10, 'Window for WYS approximation of DeepWalk.')
  flags.DEFINE_integer('dim', 30, 'Rank of SVD decomposition')
  flags.DEFINE_float('neg_coef', 0.02,
                     'The coefficient lambda, scaling term (1 - A). ')
  flags.DEFINE_bool('mult_degrees', True, 'Equivalent to conducting # of random walks from each node proportional to its degree')
  flags.DEFINE_string('renorm', 'cholesky', 'Orthonormalization function to use as part of SVD algorithm.')
FLAGS = flags.FLAGS



def score_asym(ids1, ids2, u, s, v):
  batch_u = tf.gather(u, ids1)
  batch_v = tf.gather(v, ids2)
  return tf.reduce_sum(batch_u * s * batch_v, axis=1)
  
def score_sym(ids1, ids2, u, s, v):
  return (score_asym(ids1, ids2, u, s, v) + score_asym(ids2, ids1, u, s, v))/2.0


def find_k_valid(all_train_edges, holdout=0.25):

  while True:
    np.random.shuffle(all_train_edges)

    num_train = int(holdout*len(all_train_edges))
    train_edges = all_train_edges[:num_train]
    validate_edges = all_train_edges[num_train:]
    train_edges = np.concatenate([train_edges, train_edges[:, ::-1]])
    
    num_nodes = 1 + np.max(all_train_edges)
    validate_negs = np.random.randint(size=[validate_edges.shape[0]*5, 2], low=0, high=num_nodes)
    
    A = scipy.sparse.csr_matrix(
      (np.ones([len(train_edges)], dtype='float32'), (train_edges[:, 0], train_edges[:, 1]) ))
    if A.shape != (num_nodes, num_nodes):
      print('did not capture all nodes. Retrying...')
    else:
      break

  
  RANK = 256

  mult_f = tf_fsvd.make_deepwalk_mat(A, FLAGS.window, mult_degrees=FLAGS.mult_degrees, neg_sample_coef=FLAGS.neg_coef)

  _ = tf_fsvd.fsvd(mult_f, 2, n_iter=1)  # Warm-up GPU
  print('FINDING RANK')
  started = time.time()
  u, s, v = tf_fsvd.fsvd(mult_f, RANK, n_iter=20, renorm=FLAGS.renorm )

  import sklearn.metrics
  max_auc_k = (-1, -1)
  #TRY_RANKS = range(5, RANK+1, 5)
  TRY_RANKS = [4, 8, 16, 32, 64, 128, 256]
  for k in TRY_RANKS:
    pos_scores = score_sym(validate_edges[:, 0], validate_edges[:, 1], u[:, :k], s[:k], v[:, :k])
    neg_scores = score_sym(validate_negs[:, 0], validate_negs[:, 1], u[:, :k], s[:k], v[:, :k])
    
    all_score = np.concatenate([pos_scores, neg_scores], 0)
    all_y = np.concatenate([
        np.ones([len(pos_scores)], dtype='int32'),
        np.zeros([len(neg_scores)], dtype='int32'),
    ], 0)

    auc = sklearn.metrics.roc_auc_score(all_y, all_score)
    max_auc_k = max(max_auc_k, (auc, k))
  
  best_k = max_auc_k[-1]
  print('best k at %i' % best_k)
  train_time = time.time() - started
  print('  ... done FINDING RANK')
  return best_k, train_time


def main(_):
  dataset_dir = os.path.join(os.path.expanduser(FLAGS.datasets_dir), FLAGS.dataset_name)

  train_edges = np.load(os.path.join(dataset_dir, 'train.txt.npy'))
  train_edges = np.concatenate([train_edges, train_edges[:, ::-1]], axis=0)


  if FLAGS.dim <= 0:
    rank, runtime = find_k_valid(train_edges, 0.4)
  else:
    rank, runtime = FLAGS.dim, 0
  
  num_nodes = 1 + np.max(train_edges)
  A = scipy.sparse.csr_matrix(
    (np.ones([len(train_edges)], dtype='float32'), (train_edges[:, 0], train_edges[:, 1]) ))
  
  mult_f = tf_fsvd.make_deepwalk_mat(A, FLAGS.window, mult_degrees=FLAGS.mult_degrees, neg_sample_coef=FLAGS.neg_coef)

  _ = tf_fsvd.fsvd(mult_f, 2, n_iter=1)  # Warm-up GPU
  print('training')
  started = time.time()
  u, s, v = tf_fsvd.fsvd(mult_f, rank, n_iter=10, renorm=FLAGS.renorm )
  train_time = time.time() - started
  print('  ... done training')
  
  print('testing')
  if not os.path.exists('out'):
    os.makedirs('out')
  

  test_edges = np.load(os.path.join(dataset_dir, 'test.txt.npy'))
  test_neg_edges = np.load(os.path.join(dataset_dir, 'test.neg.txt.npy'))

  def eval_metric():
    pos_scores = score_sym(test_edges[:, 0], test_edges[:, 1], u, s, v)
    neg_scores = score_sym(test_neg_edges[:, 0], test_neg_edges[:, 1], u, s, v)
    import sklearn.metrics
    all_score = np.concatenate([pos_scores, neg_scores], 0)
    all_y = np.concatenate([
        np.ones([len(pos_scores)], dtype='int32'),
        np.zeros([len(neg_scores)], dtype='int32'),
    ], 0)

    return {
      'auc': sklearn.metrics.roc_auc_score(all_y, all_score),
    }
  
  eval_results = eval_metric()
  eval_results['time'] = train_time + runtime
  print(json.dumps(eval_results))


if __name__ == '__main__':
  app.run(main)
