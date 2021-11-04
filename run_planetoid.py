# CUDA_VISIBLE_DEVICES='' python3 test_planetoid.py --svd_k=100 --dropout_levels=0 --slice_on_u=0  --dataset ind.citeseer --svd_iters=20 --layers=15
# Test w VS w/o dropout

import collections
import json
import os
import pickle
import sys
import time


import numpy as np
import scipy.sparse
import sklearn.decomposition
import tensorflow as tf

from absl import app, flags

import tf_fsvd




if __name__ == '__main__':
  flags.DEFINE_bool('use_bias', False, 'If set, bias term will be used')
  flags.DEFINE_string('dataset', 'ind.cora', '')
  flags.DEFINE_string('dataset_dir', '~/data/planetoid/data/', 'Directory where dataset files live.')
  flags.DEFINE_integer('layers', 15, 'Number of layers')
  flags.DEFINE_float('wys_neg_coef', 0.05, 'WYS negative coefficient (lambda).')
  flags.DEFINE_integer('wys_window', 3, 'Context window for Watch Your Step.')
  flags.DEFINE_integer('svd_k', 100, 'Rank of SVD for the classification matrix')
  flags.DEFINE_integer('svd_iters', 10, 'Number of iterations for estimating the SVD (= "iters" hyperparameter in Alg 1 of paper).')
  flags.DEFINE_integer('dropout_levels', 1, 'If set, feature matrix will be repeated with dropout')
  flags.DEFINE_bool('slice_on_u', False, 'If set, slice on U will be taken otherwise on design matrix M')

FLAGS = flags.FLAGS



########### DATASET


def concatenate_csr_matrices_by_rows(matrix1, matrix2):
  """Concatenates sparse csr matrices matrix1 above matrix2.
  
  Adapted from:
  https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
  """
  new_data = np.concatenate((matrix1.data, matrix2.data))
  new_indices = np.concatenate((matrix1.indices, matrix2.indices))
  new_ind_ptr = matrix2.indptr + len(matrix1.data)
  new_ind_ptr = new_ind_ptr[1:]
  new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

  return scipy.sparse.csr_matrix((new_data, new_indices, new_ind_ptr))



def load_x(filename):
  return pickle.load(open(filename, 'rb'), encoding='latin1')

def read_planetoid_dataset(dataset_name='ind.cora', dataset_dir='~/data/planetoid/data/'):
  base_path = os.path.expanduser(os.path.join(dataset_dir, dataset_name))
  if not os.path.exists(os.path.expanduser(dataset_dir)):
    raise ValueError('cannot find dataset_dir=%s. Please:\nmkdir -p ~/data; cd ~/data; git clone git@github.com:kimiyoung/planetoid.git')
  edge_lists = pickle.load(open(base_path + '.graph', 'rb'))

  allx = load_x(base_path + '.allx')
  
  ally = np.array(np.load(base_path + '.ally', allow_pickle=True), dtype='float32')
 
  testx = load_x(base_path + '.tx')

  # Add test
  test_idx = list(map(int, open(base_path + '.test.index').read().split('\n')[:-1]))

  num_test_examples = max(test_idx) - min(test_idx) + 1
  sparse_zeros = scipy.sparse.csr_matrix((num_test_examples, allx.shape[1]),
                                         dtype='float32')

  allx = concatenate_csr_matrices_by_rows(allx, sparse_zeros)
  llallx = allx.tolil()
  llallx[test_idx] = testx
  #allx = scipy.vstack([allx, sparse_zeros])

  test_idx_set = set(test_idx)


  testy = np.array(np.load(base_path + '.ty', allow_pickle=True), dtype='float32')
  ally = np.concatenate(
      [ally, np.zeros((num_test_examples, ally.shape[1]), dtype='float32')],
      0)
  ally[test_idx] = testy

  num_nodes = len(edge_lists)

  # Will be used to construct (sparse) adjacency matrix.
  edge_sets = collections.defaultdict(set)
  for node, neighbors in edge_lists.items():
    for n in neighbors:
      edge_sets[node].add(n)
      edge_sets[n].add(node)  # Assume undirected.

  # Now, build adjacency list.
  adj_indices = []
  adj_values = []
  for node, neighbors in edge_sets.items():
    for n in neighbors:
      adj_indices.append((node, n))
      adj_values.append(1)

  adj_indices = np.array(adj_indices, dtype='int32')
  adj_values = np.array(adj_values, dtype='int32')
  
  adj = scipy.sparse.csr_matrix((num_nodes, num_nodes), dtype='int32')

  adj[adj_indices[:, 0], adj_indices[:, 1]] = adj_values

  return adj, llallx, ally, test_idx


########### END OF DATASET


def main(_):
  adj, allx, ally, test_idx = read_planetoid_dataset(FLAGS.dataset, dataset_dir=FLAGS.dataset_dir)

  wys_f = tf_fsvd.make_deepwalk_mat(adj, window=FLAGS.wys_window, neg_sample_coef=FLAGS.wys_neg_coef, sym_normalization=False, rank_negatives=0)
  adj = adj + scipy.sparse.eye(adj.shape[0], dtype='float32')
  adj = adj + adj.T
  adj = (adj > 0) * np.array(1.0, 'float32')
  

  u, s, v = tf_fsvd.fsvd(wys_f, 64, n_iter=10, renorm='cholesky')
  s = s ** 1
  dense_x = allx.todense()
  dense_x = np.concatenate([dense_x, (u * np.sqrt(s)).numpy(), (v * np.sqrt(s)).numpy()], axis=1)
  dense_x = sklearn.decomposition.PCA(min(min(dense_x.shape), 1000)).fit_transform(dense_x)

  # Add embeddings.
  dense_x = np.concatenate([dense_x, (u * np.sqrt(s)).numpy(), (v * np.sqrt(s)).numpy()], axis=1)
  #adj = adj + scipy.sparse.eye(adj.shape[0]) * 1.5
  d = adj.sum(axis=1)
  normalizer = scipy.sparse.diags( np.array(1/np.sqrt(d))[:, 0] )
  normed_adj = normalizer.dot(adj.dot(normalizer))


  print('start')
  rows, cols = normed_adj.nonzero()
  values = np.array(normed_adj[rows, cols], dtype='float32')[0]
  tf_adj = tf.sparse.SparseTensor(
        tf.stack([np.array(rows, dtype='int64'), np.array(cols, dtype='int64')], axis=1),
        values,
        normed_adj.shape)
  tf_x = tf.convert_to_tensor(dense_x)
  tf_X = [tf_x]
  tf_ally = tf.convert_to_tensor(ally)
  for l in range(FLAGS.layers):
    tf_X.append(tf.sparse.sparse_dense_matmul(tf_adj, tf_X[-1]))

  if FLAGS.use_bias:
    tf_X.append(tf.ones([tf_X[0].shape[0], 1]))  # Finally, column of ones.
  tfx = tf.concat(tf_X, axis=1)
  train_idx = np.arange(ally.shape[1]*20, dtype='int32')
  val_idx = np.arange(ally.shape[1]*20, ally.shape[1]*20+500, dtype='int32')

  if FLAGS.dropout_levels:
    dropped_tfx = [tfx]
    train_x_idx = [train_idx]
    for i in range(FLAGS.dropout_levels):
      dropped_tfx.append(tf.nn.dropout(tfx, rate=0.5))
      train_x_idx.append(train_idx + adj.shape[0] * (i+1))
    tfx = tf.concat(dropped_tfx, axis=0)
    train_x_idx = np.concatenate(train_x_idx, axis=0)
    train_y_idx = np.concatenate([train_idx]*(FLAGS.dropout_levels+1), axis=0)
  else:
    train_x_idx = train_idx
    train_y_idx = train_idx

  print('TRAINING')
  started = time.time()

  xpf = tf_fsvd.DenseMatrixPF(tfx)
  if FLAGS.slice_on_u:    
    u, s, v = tf_fsvd.fsvd(xpf, FLAGS.svd_k, n_iter=FLAGS.svd_iters, renorm='cholesky')
    w = tf.matmul(
        v * tf.where(s==0, tf.zeros_like(s), 1/s),
        tf.matmul(
            tf.gather(u, train_x_idx),
            tf.gather(tf_ally, train_y_idx), transpose_a=True)
    )
  
  else: 
    train_xpf = tf_fsvd.gather(xpf, train_x_idx)
    n_redundancy = 20
    svd_k = min(FLAGS.svd_k, train_xpf.shape[0] - n_redundancy - 4)
    u, s, v = tf_fsvd.fsvd(train_xpf, svd_k, n_iter=FLAGS.svd_iters, renorm='cholesky', n_redundancy=n_redundancy)
    w = tf.matmul(
        v * tf.where(s==0, tf.zeros_like(s), 1/s),
        tf.matmul(
            u,
            tf.gather(tf_ally, train_y_idx), transpose_a=True))

  train_time = time.time() - started
  print('TESTING')

  test_preds = tf.gather(xpf.dot(w), test_idx)
  val_preds = tf.gather(xpf.dot(w), val_idx)

  test_acc = (tf.argmax(test_preds, 1) == ally[test_idx].argmax(1)).numpy().mean()
  val_acc = (tf.argmax(val_preds, 1) == ally[val_idx].argmax(1)).numpy().mean()

  output = json.dumps({'test': test_acc, 'val': val_acc, 'time': train_time})
  print(output)


if __name__ == '__main__':
  app.run(main)
