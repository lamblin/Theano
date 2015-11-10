"""
Optimizations addressing the ops in nnet root directory
"""

import theano
from theano import compile, gof, tensor

from theano.gof import (local_optimizer, EquilibriumDB, ProxyDB,
                        Optimizer, TopoOptimizer, toolbox)
from theano.tensor.nnet import (
    CorrMM, CorrMM_gradInputs, CorrMM_gradWeights)
from theano.tensor.nnet.blocksparse import (
    SparseBlockGemv,
    SparseBlockOuter,
    sparse_block_gemv_inplace,
    sparse_block_outer_inplace)
from theano.tensor.nnet.abstract_conv2d import (BaseAbstractConv2d, AbstractConv2d,
                                                AbstractConv2d_gradWeights,
                                                AbstractConv2d_gradInputs)
from theano.tensor.opt import register_specialize_device
from theano.tensor import TensorType


@gof.local_optimizer([SparseBlockGemv], inplace=True)
def local_inplace_sparse_block_gemv(node):
    """
        SparseBlockGemv(inplace=False) -> SparseBlockGemv(inplace=True)
    """
    if isinstance(node.op, SparseBlockGemv) and not node.op.inplace:
        new_node = sparse_block_gemv_inplace(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_sparse_block_gemv',
                       gof.TopoOptimizer(
                           local_inplace_sparse_block_gemv,
                           failure_callback=gof.TopoOptimizer.warn_inplace),
                       60, 'fast_run', 'inplace')  # DEBUG


@gof.local_optimizer([SparseBlockOuter], inplace=True)
def local_inplace_sparse_block_outer(node):
    """
        SparseBlockOuter(inplace=False) -> SparseBlockOuter(inplace=True)
    """
    if isinstance(node.op, SparseBlockOuter) and not node.op.inplace:
        new_node = sparse_block_outer_inplace(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_sparse_block_outer',
                       gof.TopoOptimizer(
                           local_inplace_sparse_block_outer,
                           failure_callback=gof.TopoOptimizer.warn_inplace),
                       60, 'fast_run', 'inplace')  # DEBUG


# Conv opts
@local_optimizer([AbstractConv2d])
def local_abstractconv_gemm(node):
    if not isinstance(node.op, AbstractConv2d):
        return None
    img, kern = node.inputs
    if (not isinstance(img.type, TensorType) or
        not isinstance(kern.type, TensorType)):
        return None

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1]
    # By default use CorrMM
    rval = CorrMM(border_mode=node.op.border_mode,
                  subsample=node.op.subsample)(img, kern)

    return [rval]

@local_optimizer([AbstractConv2d_gradWeights])
def local_abstractconv_gradweight_gemm(node):
    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if (not isinstance(img.type, TensorType) or \
        not isinstance(topgrad.type, TensorType)):
        return None

    rval = CorrMM_gradWeights(border_mode=node.op.border_mode,
                              subsample=node.op.subsample)(img, topgrad, shape)
    if node.op.filter_flip:
        rval = rval[:, :, ::-1, ::-1]
    rval = tensor.patternbroadcast(rval, node.outputs[0].broadcastable)
    return [rval]

@local_optimizer([AbstractConv2d_gradInputs])
def local_abstractconv_gradinputs_gemm(node):
    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if (not isinstance(kern.type, TensorType) or \
        not isinstance(topgrad.type, TensorType)):
        return None

    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1]

    rval =  CorrMM_gradInputs(border_mode=node.op.border_mode,
                              subsample=node.op.subsample)(kern, topgrad,
                                                           shape)
    return [rval]

# Register CPU convolution implementation
# They are tried in a specific order so we can control
# which ones take precedence over others.
abstractconv_groupopt = theano.gof.optdb.LocalGroupDB()
abstractconv_groupopt.__name__ = "abstractconv_opts"
#TODO all below
register_specialize_device(abstractconv_groupopt, 'gpu', 'fast_compile')

# cuDNN is first, but only registered if cuDNN is available.
conv_groupopt.register('local_abstractconv_dnn', dnn.local_abstractconv_cudnn, 20,
                       'conv_dnn',
                       'gpu', 'fast_compile', 'fast_run', 'cudnn')
# The GEMM-based convolution comes last to catch all remaining cases.
# It can be disabled by excluding 'conv_gemm'.
conv_groupopt.register('local_abstractconv_gemm', local_abstractconv_gemm, 30,
                       'conv_gemm',
                       'gpu', 'fast_compile', 'fast_run')
conv_groupopt.register('local_abstractconv_gradweight_gemm',
                       local_abstractconv_gradweight_gemm, 30,
                       'conv_gemm',
                       'gpu', 'fast_compile', 'fast_run')
conv_groupopt.register('local_abstractconv_gradinputs_gemm',
                       local_abstractconv_gradinputs_gemm, 30,
                       'conv_gemm',
                       'gpu', 'fast_compile', 'fast_run')
