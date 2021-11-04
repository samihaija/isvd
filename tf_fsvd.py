
from absl import flags
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm


flags.DEFINE_bool('disable_svd_compute_cache', False,
                  'If set, disables the cache (i.e. can recompute subexpressions')
FLAGS = flags.FLAGS

class SymbolicPF:
  """Abstract class. Instances can be passed to function `fsvd`.

  An intance of (a concrete implementation of) this class would hold an implicit
  matrix `M`, such that, this class is able to multiply it with another matrix
  `m` (by implementing function `dot`).

  Attribute `T` should evaluate to a `SymbolicPF` with implicit matrix being
  transpose of `M`.

  `shape` attribute must evaluate to shape of `M`
  """

  def dot(self, m, cache=None):
    raise NotImplementedError(
      'dot: must be able to multiply (implicit) matrix by another matrix `m`.')

  @property
  def T(self):
    raise NotImplementedError(
      'T: must return instance of SymbolicPF that is transpose of this one.')

  @property
  def shape(self):
    raise NotImplementedError(
      'shape: must return shape of implicit matrix.')

  def __add__(self, other):
    if not isinstance(other, SymbolicPF):
      raise TypeError('Expected type ProductPF but instead received %s. Did you mean to wrap with leaf(.)?' % other.__class__.__name__)
    return SumPF([self, other])

  def __matmul__(self, other):
    if not isinstance(other, SymbolicPF):
      raise TypeError('Expected type ProductPF but instead received %s. Did you mean to wrap with leaf(.)?' % other.__class__.__name__)
    return ProductPF([self, other])

  def __sub__(self, other):
    if not isinstance(other, SymbolicPF):
      raise TypeError('Expected type ProductPF but instead received %s. Did you mean to wrap with leaf(.)?' % other.__class__.__name__)
    return SumPF([self, TimesScalarPF(-1, other)])
  
  def __mul__(self, scalar):
    if not np.isscalar(scalar):
      raise TypeError('Can only multiply with scalars. For matrices, use @. For automatic broadcasting, please consider extending framework')
    return TimesScalarPF(scalar, self)

  def __rmul__(self, scalar):
    if not np.isscalar(scalar):
      raise TypeError('Can only multiply with scalars. For matrices, use @. For automatic broadcasting, please consider extending framework')
    return TimesScalarPF(scalar, self)

  def __pow__(self, integer):
    if not isinstance(integer, int):
      print('WARNING: power must be integer. Converting to int() on your behalf')
    return ProductPF([self] * int(integer))
    
  def __repr__(self):
    return "Symbolic (type %s) of shape %s" % (self.__class__.__name__, self.shape)



## Functional TF implementation of Truncated Singular Value Decomposition
# The algorithm is based on Halko et al 2009 and their recommendations, with
# some ideas adopted from code of scikit-learn.
def fsvd(fn, k, n_redundancy=None, n_iter=10, renorm='qr', verbose=None):
  """Functional TF Randomized SVD based on Halko et al 2009

  Args:
    fn: Instance of a class implementing SymbolicPF. Should hold implicit matrix
      `M` with (arbitrary) shape. Then, it must be that `fn.shape == (r, c)`,
      and `fn.dot(M1)` where `M1` has shape `(c, s)` must return `M @ M1` with
      shape `(r, s)`. Further, `fn.T.dot(M2)` where M2 has shape `(r, h)` must
      return `M @ M2` with shape `(c, h)`.
    k: rank of decomposition. Returns (approximate) top-k singular values in S
      and their corresponding left- and right- singular vectors in U, V, such
      that, `tf.matmul(U * S, V, transpose_b=True)` is the best rank-k
      approximation of matrix `M` (implicitly) stored in `fn`.
    n_redundancy: rank of "randomized" decomposition of Halko. The analysis of
      Halko provides that if n_redundancy == k, then the rank-k SVD approximation
      is, in expectation, no worse (in frobenius norm) than twice of the "true"
      rank-k SVD compared to the (implicit) matrix represented by fn.
      However, n_redundancy == k is too slow when k is large. Default sets it
      to min(k, 30).
    n_iter: Number of iterations. >=4 gives good results (with 4 passes over the
      data). We set to 10 (slower than 4) to ensure close approximation accuracy.
      The error decays exponentially with n_iter.
  Returns:
    U, s, V, s.t. tf.matmul(U*s, V, transpose_b=True) is a rank-k approximation
    of fn.
  """
  if n_redundancy is None:
    n_redundancy = min(k, 30)
  n_random = k + n_redundancy
  n_samples, n_features = fn.shape
  transpose = n_samples < n_features
  if transpose:
    # This is faster
    fn = fn.T

  Q = tf.random.normal(shape=(fn.shape[1], n_random))
  iterations = range(n_iter)
  if verbose:
    iterations = tqdm.tqdm(iterations, desc='SVD')
    print('Starting with Q of shape %s and rank %i' % (
       str(Q.shape), np.linalg.matrix_rank(Q.numpy())
    ))
  for i in iterations:
    if FLAGS.disable_svd_compute_cache:
      Q = fn.dot(_orthonormalize(Q, alg=renorm))
      Q = fn.T.dot(_orthonormalize(Q, alg=renorm))
    else:
      Q = fn.dot(_orthonormalize(Q, alg=renorm), {})
      Q = fn.T.dot(_orthonormalize(Q, alg=renorm), {})
  
  if verbose:
    print('SVD: Final step, followed by SVD on small rank matrix.')
  Q = _orthonormalize(fn.dot(Q, {}), alg=renorm)
  B = tf.transpose(fn.T.dot(Q, {}))
  s, Uhat, V = tf.linalg.svd(B)
  del B
  U = tf.matmul(Q, Uhat)

  U, V = _sign_correction(u=U, v=V, u_based_decision=not transpose)

  if transpose:
    return V[:, :k], s[:k], U[:, :k]
  else:
    return U[:, :k], s[:k], V[:, :k]


def _orthonormalize(Q, alg='qr'):
  if alg == 'qr':
    return tf.linalg.qr(Q)[0]
  elif alg == 'l2':
    return tf.math.l2_normalize(Q, axis=0)
  elif alg == 'gs':
    return tfp.math.gram_schmidt(Q)
  elif alg == 'cholesky':

    qn = tf.math.l2_normalize(Q, axis=0)   # *10
    qnqn = tf.matmul(qn, qn, transpose_a=True)
    try:
      ch = tf.linalg.cholesky(qnqn)
    except:
      raise ValueError('It seems that your matrix has an inherit rank of %i -- you may only do SVD of that rank including the redundancy parameter.' %
                             np.linalg.matrix_rank(qnqn.numpy()))

    return tf.matmul(qn, tf.linalg.inv(ch), transpose_b=True)


def _sign_correction(u, v, u_based_decision=True):
    M = u if u_based_decision else v
    max_abs_cols = tf.argmax(tf.abs(M), axis=0)
    signs = tf.sign(tf.gather_nd(M, tf.stack([max_abs_cols, tf.range(M.shape[1], dtype=tf.int64)], axis=1)))
    
    return u*signs, v*signs

# End of: Functional TF implementation of Truncated Singular Value Decomposition
## 



#### SymbolicPF implementations.

class GatherRowsPF(SymbolicPF):

  def __init__(self, pf, rows, T=None):
    self.pf = pf
    self.rows = np.array(rows, dtype='int32')
    self._t = T
  
  def dot(self, m, cache=None):
    return tf.gather(self.pf.dot(m, cache), self.rows)

  @property
  def T(self):
    if self._t is None:
      self._t = GatherColumnsPF(self.pf.T, self.rows, T=self)
    return self._t

  @property
  def shape(self):
    return (len(self.rows), self.pf.shape[1])
  

class GatherColumnsPF(SymbolicPF):

  def __init__(self, pf, cols, T=None):
    self.cols = cols
    self.pf = pf
    self._t = T
  
  def dot(self, m, cache=None):
    m_padded = tf.tensor_scatter_nd_add(
      tf.zeros((self.pf.shape[1], m.shape[1]), dtype=m.dtype),
      np.array([self.cols]).T, m)
    return self.pf.dot(m_padded, cache)


  @property
  def T(self):
    if self._t is None:
      self._t = GatherRowsPF(self.pf.T, self.cols, T=self)
    return self._t

  @property
  def shape(self):
    return (self.pf.shape[0], len(self.cols))
  


class GeoSumPF(SymbolicPF):

  def __init__(self, pf, coefs, T=None):
    self.pf = pf
    self.coefs = coefs
    assert len(self.coefs) > 0
    self._t = T
  
  def dot(self, mat, cache=None):
    power = mat
    geo_sum = 0
    for i in range(len(self.coefs)):
      power = self.pf.dot(mat, cache)
      geo_sum += tf.reduce_sum(self.coefs[i:]) * power
    return geo_sum
  
  @property
  def shape(self):
    return self.pf.shape

  @property
  def T(self):
    if self._t is None:
      self._t = GeoSumPF(self.pf.T, self.coefs, T=self)
    return self._t



class SparseMatrixPF(SymbolicPF):
  """The "implicit" matrix comes directly from a scipy.sparse.csr_matrix

  This is the most basic version: i.e., this really only extends TensorFlow to
  run "sparse SVD" on a matrix. The given `scipy.sparse.csr_matrix` will be
  converted to `tf.sparse.SparseTensor`.
  """

  def __init__(self, csr_mat=None, precomputed_tfs=None, T=None):
    """Constructs matrix from csr_mat (or alternatively, tf.sparse.tensor).

    Args:
      csr_mat: instance of scipy.sparse.csr_mat (or any other sparse matrix
        class). This matrix will only be read once and converted to
        tf.sparse.SparseTensor.
      precomputed_tfs: (optional) matrix (2D) instance of tf.sparse.SparseTensor.
        if not given, will be initialized from `csr_mat`.
      T: (do not provide) if given, must be instance of SymbolicPF with implicit
        matrix as the transpose of this one. If not provided (recommended) it
        will be automatically (lazily) computed.
    """
    super().__init__()
    if precomputed_tfs is None and csr_mat is None:
      raise ValueError('Require at least one of csr_mat or precomputed_tfs')
    if precomputed_tfs is None:
      rows, cols = csr_mat.nonzero()
      values = np.array(csr_mat[rows, cols], dtype='float32')[0]
      precomputed_tfs = tf.sparse.SparseTensor(
        tf.stack([np.array(rows, dtype='int64'), np.array(cols, dtype='int64')], axis=1),
        values,
        csr_mat.shape)
   
    self._shape = precomputed_tfs.shape
    self.csr_mat = csr_mat
    self.tfs = precomputed_tfs  # tensorflow sparse tensor.
    self._t = T

  def dot(self, v, cache=None):
    return tf.sparse.sparse_dense_matmul(self.tfs, v)

  @property
  def T(self):
    """Returns SymbolicPF with implicit matrix being transpose of this one."""
    if self._t is None:
      self._t = SparseMatrixPF(
            self.csr_mat.T if self.csr_mat is not None else None,
            precomputed_tfs=tf.sparse.transpose(self.tfs),
            T=self)
    
    return self._t

  @property
  def shape(self):
    return self._shape


class BlockWisePF(SymbolicPF):
  """Product that concatenates, column-wise, one or more (implicit) matrices.

  Constructor takes one or more SymbolicPF instances. All of which must contain
  the same number of rows (e.g., = r) but can have different number of columns
  (e.g., c1, c2, c3, ...). As expected, the resulting shape will have the same
  number of rows as the input matrices and the number of columns will is the sum
  of number of columns of input (shape = (r, c1+c2+c3+...)).
  """

  def __init__(self, fns, T=None, concat_axis=1):
    """Concatenate (implicit) matrices stored in `fns`, column-wise.

    Args:
      fns: list. Each entry must be an instance of class implementing SymbolicPF.
      T: (do not provide) if given, must be instance of SymbolicPF with implicit
        matrix as the transpose of this one. If not provided (recommended) it
        will be automatically (lazily) computed.
      concat_axis: fixed to 1 (i.e. concatenates column-wise).
    """
    self.fns = fns
    self._t = T
    self.concat_axis = concat_axis

  @property
  def shape(self):
    size_other_axis = self.fns[0].shape[1 - self.concat_axis]
    for fn in self.fns[1:]:
      assert fn.shape[1 - self.concat_axis] == size_other_axis
    total = np.sum([fn.shape[self.concat_axis] for fn in self.fns])
    myshape = [0, 0]
    myshape[self.concat_axis] = total
    myshape[1 - self.concat_axis] = size_other_axis
    return tuple(myshape)
  
  def dot(self, v, cache=None):
    assert self.shape[1] == v.shape[0]
    if self.concat_axis == 0:
      dots = [fn.dot(v, cache) for fn in self.fns]
      return tf.concat(dots, axis=self.concat_axis)
    else:
      dots = []
      offset = 0
      for fn in self.fns:
        fn_columns = fn.shape[1]
        dots.append(fn.dot(v[offset:offset+fn_columns]))
        offset += fn_columns
      return tf.reduce_sum(dots, axis=0)

  @property
  def T(self):
    """Returns SymbolicPF with implicit matrix being transpose of this one."""
    if self._t is None:
      fns_T = [fn.T for fn in self.fns]
      self._t = BlockWisePF(fns_T, T=self, concat_axis=1 - self.concat_axis)
    return self._t


class DenseMatrixPF(SymbolicPF):
  """Product function where implicit matrix is Dense tensor.

  On its own, this is not needed as one could just run tf.linalg.svd directly
  on the implicit matrix. However, this is useful when a dense matrix to be
  concatenated (column-wise) next to SparseMatrix (or any other implicit matrix)
  implementing SymbolicPF.
  """

  def __init__(self, m, T=None):
    """
    Args:
      m: tf.Tensor (dense 2d matrix). This will be the "implicit" matrix.
      T: (do not provide) if given, must be instance of SymbolicPF with implicit
        matrix as the transpose of this one. If not provided (recommended) it
        will be automatically (lazily) computed.
    """
    assert len(m.shape) == 2
    self.m = tf.convert_to_tensor(m)
    self._t = T

  def dot(self, v, cache=None):
    return tf.matmul(self.m, v)
  
  @property
  def shape(self):
    return self.m.shape

  @property
  def T(self):
    """Returns SymbolicPF with implicit matrix being transpose of this one."""
    if self._t is None:
      self._t = DenseMatrixPF(tf.transpose(self.m), T=self)
    return self._t


class SumPF(SymbolicPF):

  def __init__(self, pfs, T=None):
    self.pfs = pfs
    for pf in pfs:
      assert pf.shape == pfs[0].shape
    self._t = T

  def dot(self, m, cache=None):
    sum_ = self.pfs[0].dot(m, cache)
    for pf in self.pfs[1:]:
      sum_ += pf.dot(m, cache)
    return sum_

  @property
  def T(self):
    if self._t:
      return self._t

    self._t = SumPF([pf.T for pf in self.pfs], T=self)
    return self._t

  @property
  def shape(self):
    return self.pfs[0].shape


class TimesScalarPF(SymbolicPF):

  def __init__(self, scalar, pf):
    self.scalar = scalar
    self.pf = pf
    
  def dot(self, m, cache=None):
    return self.pf.dot(m, cache) * self.scalar

  @property
  def shape(self):
    return self.pf.shape

  @property
  def T(self):
    return self

class ProductPF(SymbolicPF):

  def __init__(self, pfs, T=None):
    self.pfs = pfs
    for i in range(len(pfs) - 1):
      assert pfs[i].shape[1] == pfs[i+1].shape[0]
    self._t = T

  @property
  def shape(self):
    return (self.pfs[0].shape[0], self.pfs[-1].shape[1])

  @property
  def T(self):
    if self._t is None:
      self._t = ProductPF([pf.T for pf in reversed(self.pfs)], T=self)
    return self._t

  def dot(self, m, cache=None):
    product = m
    for i, pf in enumerate(reversed(self.pfs)):
      if cache is not None:
        cache_key = tuple(self.pfs[-(i+1):] + [m._id])
        if cache_key in cache:
          product = cache[cache_key]
        else:
          product = pf.dot(product, cache)
          cache[cache_key] = product
      else:
        product = pf.dot(product, cache)
    return product


class DiagPF(SymbolicPF):

  def __init__(self, diagonal_vector):
    assert len(diagonal_vector.shape) == 1
    self._colvec = tf.expand_dims(diagonal_vector, 1)

  @property
  def shape(self):
    dim = self._colvec.shape[0]
    return (dim, dim)

  @property
  def T(self):
    return self

  def dot(self, m, cache=None):
    return m * self._colvec



def make_deepwalk_mat(
    csr_adj, window=5, Q=None, mult_degrees=False, sym_normalization=True,
    neg_sample_coef=0.02, rank_negatives=0):

  degrees = np.array(csr_adj.sum(axis=1), dtype='float32')[:, 0]
  degrees = np.clip(degrees, 1, None)
  if sym_normalization:
    sqrt_degrees = np.sqrt(degrees)
    degrees = sqrt_degrees
    inv_sqrt_degrees = scipy.sparse.diags(1.0/sqrt_degrees)
    csr_normalized = inv_sqrt_degrees.dot(csr_adj).dot(inv_sqrt_degrees)
  else:
    inv_degrees = scipy.sparse.diags(1.0/degrees)
    csr_normalized = inv_degrees.dot(csr_adj)

  tt = leaf(csr_normalized)

  if Q is None:
    # Default of deepwalk per WYS
    Q = window - np.arange(window, dtype='float32')
    Q /= np.sum(Q)
  else:
    window = len(Q)

  tm = sum([q * tt ** (j+1) for j, q in enumerate(Q)])
  if mult_degrees:
    tm = tm @ DiagPF(degrees)
  

  # Negatives.
  trow1 = leaf(tf.ones([1, csr_adj.shape[1]]))
  if rank_negatives == 1:
    tm = tm - trow1.T @ trow1 * neg_sample_coef
  elif rank_negatives:
    SVDA = scipy.sparse.linalg.svds(csr_adj, k=rank_negatives)
    SVDA_US = SVDA[0] * SVDA[1]
    SVDA_VS = SVDA[2].T * SVDA[1]
    tm = tm - (trow1.T @ trow1 - leaf(SVDA_US) @ leaf(SVDA_VS).T) * neg_sample_coef
  else:  # rank_negatives is 0 or None. Do full-rank negatives.
    ta = leaf(csr_adj)
    tm = tm - (trow1.T @ trow1 - ta) * neg_sample_coef

  return tm


def leaf(matrix):
  if scipy.sparse.issparse(matrix):
    return SparseMatrixPF(csr_mat=matrix)
  elif isinstance(matrix, tf.sparse.SparseTensor):
    return SparseMatrixPF(precomputed_tfs=matrix)
  # Otherwise, assume dense
  return DenseMatrixPF(matrix)


def sum(terms):
  return SumPF(terms)


def gather(pf, indices, axis=0):
  assert axis in (0, 1)
  if axis == 1:
    pf = pf.T
  if pf.__class__ == DenseMatrixPF:
    return DenseMatrixPF(tf.gather(pf.m, indices))
  return GatherRowsPF(pf, indices)



def test_rsvdf():
  import scipy.sparse as sp
  M = sp.csr_matrix((50, 100))
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
      if (i+j) % 2 == 0:
        M[i, j] = i + j

  u,s,v = fsvd(SparseMatrixPF(M), 4)
  assert np.all(np.abs(M.todense() - tf.matmul(u*s, v, transpose_b=True).numpy()) < 1e-3)

  M = M.T
  u,s,v = fsvd(SparseMatrixPF(M), 4)
  assert np.all(np.abs(M.todense() - tf.matmul(u*s, v, transpose_b=True).numpy()) < 1e-3)
  
  print('Test passes.')

if __name__ == '__main__':
  test_rsvdf()
