import warnings

import theano
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp

from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.sandbox.cuda import cuda_ndarray

cula_available = False
dimshuffle = cuda_ndarray.cuda_ndarray.dimshuffle

try:
    from scikits.cuda import cula
    cula_available = True
except (ImportError, OSError):
    pass

cula_initialized = False


class GpuSolve(GpuOp):
    """
    CULA GPU solver OP.

    trans: Whether to take the transpose of the input matrix
    or not.
    """
    def __init__(self, trans='N'):
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

        # Initialize CULA the first time it is needed
        global cula_initialized
        if cula_available and cula and not cula_initialized:
            cula.culaInitialize()
            cula_initialized = True

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            # size of the matrices to invert
            z = outputs[0]

            # Matrix
            A = inputs[0][0]

            # Solution vectors
            b = inputs[1][0]

            # A is not explicitly converted between C and F order, instead we
            # switch the "transpose" flag
            if self.trans in ('T', 'C'):
                trans = 'N'
            else:
                trans = 'T'

            # Convert b to F-order from c-order.
            b_cpy = dimshuffle(b, (1, 0)).reshape((b.shape[0], b.shape[1]))

            # This copy forces allocation of a new C-contiguous buffer
            # and returns it.
            A_cpy = A.copy()
            b_cpy = b_cpy.copy()

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

            A_pycuda, b_pycuda = cula_gpu_solve(A_pycuda, b_pycuda, trans)

            # Convert b to F-order from c-order and assign it to output:
            b_cpy = b_cpy.reshape(b.shape[::-1])
            b_cpy = dimshuffle(b_cpy, (1, 0))
            z[0] = b_cpy

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

gpu_solve = GpuSolve()
