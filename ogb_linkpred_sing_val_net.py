# python3 ogb_linkpred_sing_val_net.py --lr 1e-2 --epochs 1

import os
import sys
import time
import json

from absl import app, flags
from ogb.linkproppred import LinkPropPredDataset, Evaluator
import numpy as np
import scipy.sparse
import tensorflow as tf
import tqdm

import tf_fsvd

flags.DEFINE_integer('epochs', 1,' Number of finetune epochs')
flags.DEFINE_string('dataset', 'ogbl-ddi', '')
flags.DEFINE_integer('hits', 20, 'DDI is evalued with hits@20')
flags.DEFINE_boolean('hinge', False,' If set, hinge loss will be used for finetuning')
flags.DEFINE_integer('wys_window', 5, 'Max power of transition matrix for WYS.')
flags.DEFINE_float('wys_neg_coef', 1, 'Negative co-efficient for WYS.')
flags.DEFINE_integer('svd_iters', 7, 'Number of svd iterations. Higher is better. 10 is usually close to perfect SVD.')
flags.DEFINE_float('lr', 1e-2, 'Learning rate')
flags.DEFINE_integer('k', 100, 'Rank of SVD.')
flags.DEFINE_bool('sym', False, 'If set, uses symmetric T')
flags.DEFINE_integer('early_stop_steps', 0, '')
flags.DEFINE_bool('ipython', False, 'If set, trigers IPython at the end')
flags.DEFINE_integer('num_runs', 1, 'number of training runs (to average accuracy)')
flags.DEFINE_string('renorm', 'cholesky', 'Orthonormalization function to use as part of SVD algorithm.')
FLAGS = flags.FLAGS




class SingularValueModel:

  @property
  def trainable_variables(self):
    return []

  def clip(self):
    pass

class TrainableSModel(SingularValueModel):

  def __init__(self, u, s, v):
    self.u, self.s, self.v = u, s , v
    self.ts = tf.Variable(tf.zeros_like(s))
    self.ts.assign(tf.math.log(s))
    self.tpow = tf.Variable(np.array(1.0, dtype='float32'))

  def set_basis(self, u, s, v):
    self.u = u
    self.v = v
    self.s = s
  
  def _score_asym(self, ids1, ids2, u, s, v):
    batch_u = tf.gather(u, ids1)
    batch_v = tf.gather(v, ids2)
    return tf.reduce_sum(batch_u * s * batch_v, axis=1)
  
  def _score_sym(self, ids1, ids2, u, s, v):
    return (self._score_asym(ids1, ids2, u, s, v) + self._score_asym(ids2, ids1, u, s, v))/2.0

  @property
  def trainable_variables(self):
    return [self.ts, self.tpow]

  def score(self, ids1, ids2):
    overflow = self.u.shape[0]
    softmax_s = overflow * tf.nn.softmax(self.ts) ** self.tpow
    return self._score_sym(ids1, ids2, self.u, softmax_s, self.v)



class CvxPowSModel2(TrainableSModel):
  
  def __init__(self, u, s, v):
    self.noop = False
    self.act_converged = False
    D = 0.005
    self.pows = np.arange(0.5, 2.0+D, D)
    self.set_basis(u, s, v)
    #self.M = tf.Variable(np.array(0.9, dtype='float32'))  # (volume) multiplier 
    self.M = tf.Variable(np.array(1, dtype='float32'))  # (volume) multiplier 
    self.b = tf.Variable(np.array(0, dtype='float32'))  # bias
    
    self.tpows = tf.Variable(np.array(self.pows, dtype='float32'))
    self.center = tf.Variable(np.array(1, dtype='float32'))
    self.logscale = tf.Variable(np.array(5, dtype='float32'))


  def set_basis(self, u, s, v):
    super().set_basis(u, s, v)
    self.s_pows = []
    for p in self.pows:
      s_pow = self.s ** p
      self.s_pows.append(s_pow)

    self.stacked_pows = tf.stack(self.s_pows, axis=1)


  @property
  def trainable_variables(self):
    return [self.center, self.logscale]#, self.M] # ,self.b
  
  def score(self, ids1, ids2):
    equiv_s = self.s
    if self.act_converged:
      return self._score_sym(ids1, ids2, self.u, self.s**self.center, self.v)

    if not self.noop:
      cvx_weights = tf.nn.softmax(-(self.center - self.tpows)**2
                  * tf.math.exp(self.logscale)
                  #* self.logscale ** 2
                  )
      cvx_sum_s = self.stacked_pows * cvx_weights
      cvx_sum_s = (self.M**2) * tf.reduce_sum(cvx_sum_s, axis=1) + self.b
      equiv_s = cvx_sum_s
    return self._score_sym(ids1, ids2, self.u, equiv_s, self.v)

  def clip(self):
    self.logscale.assign(tf.clip_by_value(self.logscale, 0, 20))
    self.center.assign(tf.clip_by_value(self.center, self.pows[2], self.pows[-3]))

  def __str__(self):
    return str('m=%g s=%g b=%g M=%g' % (self.center.numpy(),
                     np.exp(self.logscale.numpy()),
                     #self.logscale.numpy()**2,
                     self.b.numpy(),
                     self.M.numpy()**2,
                     )  )


def main(_):
  ds = LinkPropPredDataset(FLAGS.dataset)
  split_edge = ds.get_edge_split()
    
  dataset = LinkPropPredDataset(FLAGS.dataset)
  evaluator = Evaluator(name=FLAGS.dataset)
  evaluator.K = FLAGS.hits 
  
  train_edges = split_edge['train']['edge']
  val_edges = split_edge['valid']['edge']
  num_nodes = train_edges.max() + 1
  def get_USV(train_edges):
    train_edges = np.concatenate([train_edges, train_edges[:, ::-1]], axis=0)

    spa = scipy.sparse.csr_matrix((np.ones([len(train_edges)]), (train_edges[:, 0], train_edges[:, 1]) ))
    spa = (spa > 0) * np.array(1.0, dtype='float32')
    window = FLAGS.wys_window
    mult_f = tf_fsvd.make_deepwalk_mat(spa, window=FLAGS.wys_window, neg_sample_coef=FLAGS.wys_neg_coef, sym_normalization=FLAGS.sym)
    start_time = time.time()
    u, s, v = tf_fsvd.fsvd(mult_f, FLAGS.k, n_iter=FLAGS.svd_iters, n_redundancy=FLAGS.k, renorm=FLAGS.renorm, verbose=True)
    init_train_time = time.time() - start_time
    return (u, s, v), init_train_time

  (u, s, v), init_train_time = get_USV(split_edge['train']['edge'])

  train_edges = np.concatenate([train_edges, train_edges[:, ::-1]], axis=0)

  test_metrics = []
  val_metrics = []
  times = []

  model = CvxPowSModel2(u, s, v)
  
  model.noop = True
  split_edge = dataset.get_edge_split()
  
  def eval_metric(split='test'):
    pos_edges = split_edge[split]['edge']
    neg_edges = split_edge[split]['edge_neg']
    pos_scores = model.score(pos_edges[:, 0], pos_edges[:, 1]).numpy()
    neg_scores = model.score(neg_edges[:, 0], neg_edges[:, 1]).numpy()
    metric = evaluator.eval({'y_pred_pos': pos_scores, 'y_pred_neg': neg_scores})
    return metric
  
  test_metrics.append(eval_metric())
  val_metrics.append(eval_metric('valid'))
  times.append(init_train_time)
  
  print(test_metrics)
  

  trainable_vars = model.trainable_variables
  if not trainable_vars:
    print('Nothing to train. Results are final.')
    return 0
  #exit(0)
  model.noop = False
  ####
  # START OF TEST CODE
  #np.savez('out/ddi_svd', u=u.numpy(), s=s.numpy(), v=v.numpy())
  #np.save('out/ddi_s', s.numpy())
  EPOCHS = FLAGS.epochs
  BATCH_SIZE = 1000
  total_steps = 0
  opt = tf.keras.optimizers.Adam(FLAGS.lr)
  #opt = tf.keras.optimizers.SGD(FLAGS.lr)

  for i in range(EPOCHS):
    edge_ids = np.random.permutation(train_edges.shape[0])
    losses = []
    tt = tqdm.tqdm(range(0, len(edge_ids), BATCH_SIZE))
    epoch_start_time = time.time()
    for starti in tt:
      if FLAGS.early_stop_steps > 0 and total_steps > FLAGS.early_stop_steps:
        break
      total_steps += 1
      endi = starti + BATCH_SIZE
      if endi > len(edge_ids):
        continue
      #
      indices = edge_ids[starti:endi]

      pos_edges = train_edges[indices]
      NEGK = 10
      neg_edges = np.random.choice(num_nodes, size=(NEGK*pos_edges.shape[0], 2))
      
      # Fixed.
      with tf.GradientTape() as tape:
        pos_scores = model.score(pos_edges[:, 0], pos_edges[:, 1])
        neg_scores = model.score(neg_edges[:, 0], neg_edges[:, 1])
        all_scores = tf.concat([pos_scores, neg_scores], axis=0)
        if FLAGS.hinge:
          #import IPython; IPython.embed()
          poss = tf.math.log_sigmoid(pos_scores)
          negs = tf.math.log_sigmoid(neg_scores)
          all_scores = tf.concat([poss, negs], axis=0)
          all_labels = tf.concat([tf.ones_like(pos_scores), tf.zeros_like(neg_scores)-1], axis=0)
          loss = tf.keras.losses.hinge(y_true=all_labels, y_pred=all_scores) 
          loss = tf.reduce_mean(loss)
        else:
          all_labels = tf.concat([tf.ones_like(pos_scores), tf.zeros_like(neg_scores)], axis=0)
          loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=all_labels, logits=all_scores))
          #loss += 1e-4 * tf.reduce_sum([tf.reduce_sum(v**2) for v in trainable_vars])
        losses.append(loss.numpy().mean())
      
      grads = tape.gradient(loss, trainable_vars)
      opt.apply_gradients(zip(grads, trainable_vars))
      model.clip()
      tt.set_description(str(model))
      ### EVAL
    epoch_train_time = time.time() - epoch_start_time
    test_metrics.append(eval_metric())
    val_metrics.append(eval_metric('valid'))
    times.append(epoch_train_time + times[-1])
    print('@%i] loss=%g ; eval=%s, model=%s' % (
        i, np.mean(losses), str(test_metrics[-1]), str(model) ))
    if FLAGS.early_stop_steps > 0 and total_steps > FLAGS.early_stop_steps:
      break

  # Compute
  model.act_converged = True

  # END OF TEST CODE
  ####
  #

  if True:
    updated_usv, update_time = get_USV(np.concatenate([train_edges, val_edges], axis=0))
    model.set_basis(*updated_usv)
    test_metrics.append(eval_metric())
    times.append(update_time + times[-1])
  
  print(json.dumps({'test': test_metrics, 'times': times}))
 
  if FLAGS.ipython:
    import IPython; IPython.embed()


if __name__ == '__main__':
  app.run(main)
