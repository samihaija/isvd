# for i in {1..10}; do python3 test_arxiv_mixed_device.py --finetune_device='gpu:0' --train_loop training_loops/finetune_arxiv_final.json; done

import json
import time
import os


from absl import app, flags
import numpy as np
from ogb.nodeproppred import NodePropPredDataset
import scipy.sparse
import tensorflow as tf
import tqdm

import tf_fsvd
import gttf.framework.compact_adj
import gttf.framework.traversals
import gttf.framework.accumulation
import gttf.utils.tf_utils


flags.DEFINE_string('dataset', 'ogbn-arxiv', '')
flags.DEFINE_integer('layers', 2, '')
flags.DEFINE_string('fanouts', '', 'If given, must be ints seperated with "x"')
flags.DEFINE_integer('inv_rank', 250, 'Rank for the inverse')
flags.DEFINE_integer('svd_iter', 5, 'Rank for the inverse')
flags.DEFINE_string('renorm', 'cholesky', 'Renorm step in fsvd')
flags.DEFINE_integer('label_reuse', 1, '')
flags.DEFINE_integer('pca_x', 100, 'If set, runs PCA on X.')
flags.DEFINE_boolean('layernorm', False, 'If set, uses layernorm')
flags.DEFINE_boolean('delta', False, '')
#flags.DEFINE_integer('svd_wys', 0, 'If >0, appends WYS svd to x vector')
flags.DEFINE_boolean('val_as_train', False, 'If set, validation will be added to training.')
flags.DEFINE_integer('dropout_levels', 2, 'If set, feature matrix will be repeated with dropout')
flags.DEFINE_float('scale_y', 1.0, '')
flags.DEFINE_boolean('y_dropout', False, 'If set, dropout on labels (for label-reuse) will be applied')
flags.DEFINE_string('svd_device', 'cpu:0', 'Device for executing SVD')
flags.DEFINE_string('finetune_device', 'cpu:0', 'Device for executing SVD')
flags.DEFINE_string('train_loop', 'training_loops/finetune_arxiv_final.json', 'Either JSON-encoded string or .json filename')
flags.DEFINE_boolean('ipython', False, '')

flags.DEFINE_float('l2reg', 1e-6, '')
flags.DEFINE_boolean('stochastic_eval', False, 'if set, eval will be done using GTTF')

FLAGS = flags.FLAGS


def get_all_layers(tf_adj, tf_x, tf_trainy, adj_diagonal, remove_first_layer=False, layernorm=False):
  tf_X = [tf_x]
  for l in range(FLAGS.layers):
    xprev = tf_X[-1]
    if l == 0 and FLAGS.label_reuse:
      # Concat labels.
      xprev = tf.concat([xprev, tf_trainy], axis=1)

    AX = tf.sparse.sparse_dense_matmul(tf_adj, xprev)
    
    if l == 0 and FLAGS.label_reuse:
      # prevent leakage. Remove tf_trainy from each node's row.
      AX = AX - (tf.concat([tf.zeros_like(tf_x), tf_trainy], axis=1) * tf.expand_dims(adj_diagonal, 1))

    if FLAGS.delta:
      AX = tf.concat([
        AX,
        AX - xprev,
      ], axis=1)

    tf_X.append(AX)

  if remove_first_layer:
    tf_X[0] = tf.zeros_like(tf_X[0])

  X = tf.concat(tf_X, axis=1)
  if layernorm:
    X = tf.math.l2_normalize(X, axis=1)
  return X


def main(_):
  if FLAGS.train_loop.endswith('.json'):
    TRAIN_LOOP = json.loads(open(FLAGS.train_loop).read())
  else:
    TRAIN_LOOP = json.loads(FLAGS.train_loop)

  for k, v in TRAIN_LOOP.get('flags', {}).items():
    setattr(FLAGS, k, v)

  
  
  dataset = NodePropPredDataset(name=FLAGS.dataset)
  split_idx = dataset.get_idx_split()
  train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
  if FLAGS.val_as_train:
    train_idx = np.concatenate([train_idx, valid_idx], axis=0)
  graph, label = dataset[0] # graph: library-agnostic graph object

  edges = graph['edge_index'].T
  num_nodes = edges.max() + 1

  edges = np.concatenate([
      edges,
      # Add transpose
      edges[:, ::-1]])

  A = scipy.sparse.csr_matrix(
      (np.ones([len(edges)], dtype='float32'), (edges[:, 0], edges[:, 1]) ))

  A += scipy.sparse.eye(A.shape[0])

  # Remove double-edges
  A = (A > 0) * np.array(1.0, dtype='float32')

  inv_degrees = np.array(1.0 / A.sum(0), dtype='float32')[0]
  # Symmetric normalization
  normalizer = scipy.sparse.diags(np.sqrt(inv_degrees))

  Ahat = normalizer.dot(A).dot(normalizer)
  DA = normalizer.dot(normalizer).dot(A)  # unused

  rows, cols = Ahat.nonzero()
  values = np.array(Ahat[rows, cols], dtype='float32')[0]
  num_labels = label.max() + 1
  with tf.device(FLAGS.svd_device):
    tf_adj = tf.sparse.SparseTensor(
          tf.stack([np.array(rows, dtype='int64'), np.array(cols, dtype='int64')], axis=1),
          values,
          Ahat.shape)
    tf_ally = tf.one_hot(label[:, 0], num_labels)
    np_ally = tf_ally.numpy()
    np_trainy = np_ally + 0
    np_trainy[test_idx] = 0
    tf_trainy = tf.convert_to_tensor(np_trainy)  # Oops.

  dense_x = graph['node_feat']

  import sklearn.decomposition
  
  if FLAGS.pca_x:
    dense_x = sklearn.decomposition.PCA(FLAGS.pca_x).fit_transform(dense_x)
  with tf.device(FLAGS.svd_device):
    tf_x = tf.convert_to_tensor(dense_x)
  
    X = get_all_layers(tf_adj, tf_x, tf_trainy, inv_degrees, layernorm=FLAGS.layernorm)

    X2 = get_all_layers(tf_adj, tf_x, tf.zeros_like(tf_trainy), inv_degrees, layernorm=FLAGS.layernorm)

    xgroups = [
      X, X2,
      #X3, X4
      ]
    for l in range(FLAGS.dropout_levels):
      xgroups.append(tf.nn.dropout(xgroups[0], rate=0.5))
    XC = tf_fsvd.leaf(tf.concat(xgroups, axis=0))

    svdX = tf_fsvd.fsvd(
        XC, FLAGS.inv_rank, n_iter=FLAGS.svd_iter, renorm=FLAGS.renorm, verbose=True,
        )
    
    
    combined_idx = [train_idx + A.shape[0]*ii for ii in range(2+FLAGS.dropout_levels)]
    combined_idx = tf.concat(combined_idx, axis=0)


  def run_test(U, S, V, Y, scale_y=None):
    SCALE_Y = scale_y or FLAGS.scale_y
    W = tf.matmul(
        V * (1/S),
        tf.matmul(U, Y*SCALE_Y - (SCALE_Y/2), transpose_a=True))
    Wtest = W

    scores = tf.gather(XC.dot(Wtest), test_idx)
    for i in range(FLAGS.label_reuse):
      #
      np_combined_y = tf_trainy.numpy()
      np_combined_y[test_idx] = tf.nn.softmax(scores * 10000).numpy()
      tf_combined_y = tf.convert_to_tensor(np_combined_y)
      testX = get_all_layers(tf_adj, tf_x, tf_combined_y, inv_degrees, layernorm=FLAGS.layernorm)
      scores = tf.matmul(testX, Wtest)
      scores = tf.gather(scores, test_idx)
    

    # Test accuracy
    ypred = tf.argmax(scores, axis=1)
    ytrue = tf.argmax(tf.gather(tf_ally, test_idx),  axis=1)
    accuracy = tf.reduce_mean(tf.cast(ypred == ytrue, tf.float32))
    print('test accuracy', accuracy.numpy())
    return W, float(accuracy.numpy())


  with tf.device(FLAGS.svd_device):
    W, svd_test_accuracy = run_test(U=tf.gather(svdX[0], combined_idx), S=svdX[1], V=svdX[2], Y=tf.gather(tf_ally, tf.concat([train_idx]*(2+FLAGS.dropout_levels), axis=0)))

  # Compact Adjacency
  cadj_fname = os.path.join('dataset', FLAGS.dataset + '-cadj.np.gz')
  if os.path.exists(cadj_fname):
    cadj = gttf.framework.compact_adj.CompactAdjacency.from_file(cadj_fname)
  else:
    cadj = gttf.framework.compact_adj.CompactAdjacency(A - scipy.sparse.eye(A.shape[0]))
    cadj.save(cadj_fname)



  class Net:  # With label re-use
    def __init__(self, W, dimx, dimy):
      self.trainable_variables = []
      self.w_init = W
      self.ws_init = []
      self.trainables = []
      offset = 0
      for i in range(FLAGS.layers+1):
        endi = offset + dimx
        if i > 0 and FLAGS.label_reuse:
          endi += dimy
        
        w = W[offset:endi]
        self.ws_init.append(w)
        # <SplitRelu layer at output>
        pos_layer = tf.keras.layers.Dense(w.shape[1])
        neg_layer = tf.keras.layers.Dense(w.shape[1], use_bias=False)
        pos_layer(tf.zeros([1, w.shape[0]]))
        neg_layer(tf.zeros([1, w.shape[0]]))
        pos_layer.trainable_variables[0].assign(w)
        neg_layer.trainable_variables[0].assign(-w)
        self.trainables.append([
          pos_layer,  # Positive part to output
          neg_layer, # Negative part to output
        ])
        # </SplitRelu layer at output>
        self.trainable_variables += pos_layer.trainable_variables + neg_layer.trainable_variables
        if i < FLAGS.layers:
          # <SplitRelu layer for propagation>
          pos_layer = tf.keras.layers.Dense(w.shape[0])
          neg_layer = tf.keras.layers.Dense(w.shape[0], use_bias=False)  # No need for 2 bias terms
          neg_layer(tf.zeros([1, w.shape[0]]))
          pos_layer(tf.zeros([1, w.shape[0]]))
          self.trainables[-1].append(pos_layer)
          self.trainables[-1].append(neg_layer)
          pos_layer.trainable_variables[0].assign(tf.zeros([w.shape[0], w.shape[0]]))
          neg_layer.trainable_variables[0].assign(-tf.zeros([w.shape[0], w.shape[0]]))
          self.trainable_variables += pos_layer.trainable_variables + neg_layer.trainable_variables
          # </SplitRelu layer for propagation>
        offset = endi

    def __call__(self, adj, x, y, adj_diagonal, dropout=None):
      net = x
      output = []
      for layer, layers in enumerate(self.trainables):
        # Output path
        pos_out_layer, neg_out_layer = layers[:2]
        pos_net = pos_out_layer(net)
        if dropout: pos_net = tf.nn.dropout(pos_net, rate=dropout)
        pos_net = tf.nn.relu(pos_net)
        neg_net = neg_out_layer(net)
        if dropout: neg_net = tf.nn.dropout(neg_net, rate=dropout)
        neg_net = tf.nn.relu(neg_net)
        out_net = pos_net - neg_net
        output.append(out_net)

        if layer < FLAGS.layers:  # Forward propagation path
          layer_input = net
          pos_fwd_layer, neg_fwd_layer = layers[-2:]

          pos_net = pos_fwd_layer(net) + layer_input # Residual connection
          if dropout: pos_net = tf.nn.dropout(pos_net, rate=dropout)

          neg_net = neg_fwd_layer(net) - layer_input  # Residual connection
          if dropout: neg_net = tf.nn.dropout(neg_net, rate=dropout)
          
          pos_net = tf.sparse.sparse_dense_matmul(adj, pos_net)
          neg_net = tf.sparse.sparse_dense_matmul(adj, neg_net)
          pos_net = tf.nn.relu(pos_net)
          neg_net = tf.nn.relu(neg_net)

          net = pos_net - neg_net

          if layer == 0 and FLAGS.label_reuse:
            y_columns = tf.sparse.sparse_dense_matmul(adj, y)
            y_columns -= tf.expand_dims(adj_diagonal, 1) * y
            net = tf.concat([net, y_columns], axis=1)

      return tf.reduce_sum(output, axis=0)

  with tf.device(FLAGS.finetune_device):
    net = Net(W, tf_x.shape[1], tf_ally.shape[1])
    adj_indices = tf.stack([np.array(rows, dtype='int64'), np.array(cols, dtype='int64')], axis=1)
    tf_adj = tf.sparse.SparseTensor(
          tf.stack([np.array(rows, dtype='int64'), np.array(cols, dtype='int64')], axis=1),
          values,
          Ahat.shape)
  
    att_net = tf.keras.models.Sequential([tf.keras.layers.Dense(50, use_bias=False)])

    opt = tf.keras.optimizers.Adam(1e-4)
    #opt = tf.keras.optimizers.SGD(1e-4, momentum=0.9)
    tf_x = tf_x + 0

  def finetune_gttf(num_epochs=5, eval_every=1):
    all_accuracies = []
    net_variables = None
    DROPOUT = 0.5
    BATCH_SIZE = 500
    if FLAGS.fanouts:
      FANOUT = [int(f) for f in FLAGS.fanouts.split('x')]
    else:
      FANOUT = [4] + ([2]*(FLAGS.layers-1))

    for i in tqdm.tqdm(range(num_epochs)):
      perm = np.random.permutation(train_idx)
      for starti in tqdm.tqdm(range(0, len(perm), BATCH_SIZE)):
        endi = starti + BATCH_SIZE
        if endi > len(perm):
          continue
        #
        seed_nodes = perm[starti:endi]
        walk_forest = gttf.framework.traversals.np_traverse(cadj, seed_nodes, fanouts=FANOUT)
        sampled_adj = gttf.framework.accumulation.SampledAdjacency.from_walk_forest(walk_forest, A.shape)
        batch_a = sampled_adj.tf_trimmed
        batch_a, normalizer = gttf.utils.tf_utils.kipf_renorm_tf(batch_a, return_normalizer=True)
        batch_inv_degrees = normalizer * normalizer
        batch_x = sampled_adj.tf_trim_x(tf_x)
        batch_x_y = sampled_adj.tf_trim_x(tf_trainy)

        with tf.GradientTape() as tape:
          if FLAGS.y_dropout:
            dropped_out_y = batch_x_y * tf.cast(tf.random.uniform([batch_x_y.shape[0], 1], minval=0, maxval=2, dtype=tf.dtypes.int32), tf.float32)
          else:
            dropped_out_y = batch_x_y
          h = net(batch_a, batch_x, dropped_out_y, batch_inv_degrees, dropout=DROPOUT)
          h_untrimmed = sampled_adj.tf_untrim_gather(h, seed_nodes)

          # Compute Loss.
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
              labels=tf.gather(tf_trainy, seed_nodes),
              logits=h_untrimmed,
          ))

          ##
          if net_variables is None:
            net_variables = net.trainable_variables
          trainable_variables = net_variables

          reg_loss = [tf.reduce_sum(v**2) for v in trainable_variables]
          reg_loss = FLAGS.l2reg * tf.reduce_sum(reg_loss)
          loss = tf.reduce_mean(loss) + reg_loss
          

        grads = tape.gradient(loss, trainable_variables)
        grads_and_vars = zip(grads, trainable_variables)
        opt.apply_gradients(grads_and_vars)
      

      if (i+1) % eval_every == 0:
        ### EVAL
        if FLAGS.stochastic_eval:
          test_batch = np.random.choice(test_idx, 10000)
          #test_batch = test_idx[:10000]
          seed_nodes = test_batch
          TEST_FANOUTS = [f*5 for f in FANOUT]
          walk_forest = gttf.framework.traversals.np_traverse(cadj, seed_nodes, fanouts=TEST_FANOUTS)
          sampled_adj = gttf.framework.accumulation.SampledAdjacency.from_walk_forest(walk_forest, A.shape)
          batch_a = sampled_adj.tf_trimmed
          batch_a, normalizer = gttf.utils.tf_utils.kipf_renorm_tf(batch_a, return_normalizer=True)
          batch_inv_degrees = normalizer * normalizer
          batch_x = sampled_adj.tf_trim_x(tf_x)
          batch_x_y = sampled_adj.tf_trim_x(tf_trainy)

          scores = net(batch_a, batch_x, batch_x_y, batch_inv_degrees)
          scores = sampled_adj.tf_untrim_gather(scores, test_batch)
          test_idx_locations = sampled_adj.tf_untrim_gather( tf.range(batch_x_y.shape[0]), test_batch )
          for i in range(FLAGS.label_reuse):
            updated_batch_x_y = tf.tensor_scatter_nd_add(
              batch_x_y,
              tf.expand_dims(test_idx_locations, 1),
              tf.nn.softmax(scores * 10))
            scores = net(batch_a, batch_x, updated_batch_x_y, batch_inv_degrees)
            scores = sampled_adj.tf_untrim_gather(scores, test_batch)

          # Test accuracy
          ypred = tf.argmax(scores, axis=1)
          ytrue = tf.argmax(tf.gather(tf_ally, test_batch),  axis=1)
          accuracy = tf.reduce_mean(tf.cast(ypred == ytrue, tf.float32))
          print('test accuracy', accuracy.numpy())
          all_accuracies.append((i, float(accuracy.numpy())))
        else:
          scores = net(tf_adj, tf_x, tf_trainy, inv_degrees)
          scores = tf.gather(scores, test_idx)
          for i in range(FLAGS.label_reuse):
            #
            np_combined_y = tf_trainy.numpy()
            np_combined_y[test_idx] = tf.nn.softmax(scores * 10).numpy()
            tf_combined_y = tf.convert_to_tensor(np_combined_y)
            scores = net(tf_adj, tf_x, tf_combined_y, inv_degrees)
            scores = tf.gather(scores, test_idx)
          

          # Test accuracy
          ypred = tf.argmax(scores, axis=1)
          ytrue = tf.argmax(tf.gather(tf_ally, test_idx),  axis=1)
          accuracy = tf.reduce_mean(tf.cast(ypred == ytrue, tf.float32))
          print('test accuracy', accuracy.numpy())
          all_accuracies.append((i, float(accuracy.numpy())))
    return all_accuracies

  
  FINETUNE_FN_DICT = {
    "finetune_gttf": finetune_gttf,
  }

  with tf.device(FLAGS.finetune_device):
    fn_name = TRAIN_LOOP.get('f')
    finetune_fn = FINETUNE_FN_DICT[fn_name]
    accuracy_curve = [(-1, svd_test_accuracy)]
    def run_epochs(num_epochs, learn_rate):
      print('\n\n####  Running %i epochs of %s at learning rate %g' % (num_epochs, fn_name, learn_rate))
      opt.lr.assign(learn_rate)
      accuracy_curve.extend(finetune_fn(num_epochs=num_epochs, eval_every=1))

    if FLAGS.ipython:
      import IPython; IPython.embed()

    for num_epochs, learn_rate in TRAIN_LOOP.get('curve'):
      run_epochs(num_epochs, learn_rate)
    
    if FLAGS.train_loop.endswith('.json'):
      if not os.path.exists('train_curves'):
        os.makedirs('train_curves')
      outfile = os.path.join('train_curves', '%i_%s' % (int(time.time()), os.path.basename(FLAGS.train_loop)))
      with open(outfile, 'w') as fout:
        fout.write(json.dumps({
          'curve': accuracy_curve,
          'flags': {k: getattr(FLAGS, k) for k in dir(FLAGS)},
        }))
      
      print('wrote ' + outfile)


if __name__ == '__main__':
  app.run(main)