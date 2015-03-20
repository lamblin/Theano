import warnings

import theano
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp

from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable

cula_available = False
try:
    from scikits.cuda import cula
    cula_available = False
except ImportError:
    warnings.warn("CULA import failed in theano.sandbox.cuda.cula")

cula_initialized = False
if cula_available and cula and not cula_initialized:
    try:
        cula.culaInitialize()
        cula_initialized = True
    except:
        warnings.warn("Initialization of cula failed.")


class GpuSolve(GpuOp):
    """
    CULA GPU solver OP.

    trans: Whether to take the transpose of the input matrix
    or not. By default, we will take the transpose of the
    input matrix, before feeding it into the Op. That is
    mainly, because that CULA requires inputs to be in Fortran
    order.
    """
    def __init__(self, trans='T'):
        self.trans = trans
        super(GpuSolve, self).__init__()

    def __eq__(self, other):
        return (type(other) == type(self))

    def __hash__(self):
        return hash(type(self))

    def output_type(self, inp):
        return CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_node(self, inp1, inp2):
        inp1 = as_cuda_ndarray_variable(inp1)
        inp2 = as_cuda_ndarray_variable(inp2)

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 2
        assert inp2.ndim == 2
        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def make_thunk(self,
                   node,
                   storage_map, _,
                   no_recycling=[]):
        from theano.misc.pycuda_utils import to_gpuarray

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            # size of the matrices to invert
            z = outputs[0]

            # Matrix
            A = inputs[0][0]

            # Solution vectors
            b = inputs[1][0]

            A_cpy = A.copy()
            b_cpy = b.copy()

            # Convert b to F-order from c-order.
            b_cpy = b_cpy.dimshuffle(1, 0).reshape((b.shape[0], b.shape[1]))

            A_pycuda = to_gpuarray(A_cpy)
            b_pycuda = to_gpuarray(b_cpy)

            def cula_gpu_solve(A_, b_, trans='T'):

                A_shape = A_.shape
                b_shape = b_.shape

                assert(len(A_shape) == 2)
                assert(len(b_shape) == 2)

                if trans in ['T', 'C']:
                    l, n = A_shape
                    k, m = b_shape
                    if n != k:
                        raise ValueError('A and b must be aligned.')
                elif trans in ['N']:
                    n, l = A_shape
                    k, m = b_shape
                    if l != m:
                        raise ValueError('A and b must be aligned.')
                else:
                    raise ValueError('Invalid value for trans')

                lda = max(1, n)
                ldb = max(1, n, l)

                # construct pointer arrays needed for culaDeviceSgels
                # Cula requires you to pass a pointer for A and b.
                A_ptr = A_.gpudata
                b_ptr = b_.gpudata

                cula.culaDeviceSgels(trans, n, l, m, A_ptr, lda, b_ptr, ldb)
                return A_, b_

            A_pycuda, b_pycuda = cula_gpu_solve(A_pycuda, b_pycuda, self.trans)

            # Convert b to F-order from c-order and assign it to output:
            z[0] = b_cpy.reshape((b.shape[0], b.shape[1])).dimshuffle(1, 0)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

gpu_solve = GpuSolve()
