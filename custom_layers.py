import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer
from torch.nn.parameter import UninitializedParameter
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, LEConv,ChebConv, SAGEConv, GraphConv, GravNetConv, GatedGraphConv, ResGatedGraphConv, GATConv, GATv2Conv, TransformerConv, AGNNConv, TAGConv, GINConv, GINEConv, ARMAConv, SGConv, APPNP, MFConv, RGCNConv, FastRGCNConv, SignedConv, SignedConv, DNAConv, DNAConv, GMMConv, SplineConv, NNConv, CGConv, EdgeConv, DynamicEdgeConv, XConv, PPFConv, FeaStConv, PointTransformerConv, HypergraphConv, LEConv, PNAConv, ClusterGCNConv, GENConv, GCN2Conv, PANConv, WLConv, FiLMConv, SuperGATConv, FAConv, EGConv, PDNConv, GeneralConv, HGTConv, HEATConv, HeteroConv #, LGConv
# PointConv, 
# ECConv, 



class LazyLayerNorm(LazyModuleMixin, LayerNorm):
    """
    A `LayerNorm` with lazy initialization.
    See `LayerNorm` for details:
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
    Parameters
    ----------
    - eps : float, default 1e-5
        a value added to the denominator for numerical stability.
    - elementwise_affine : bool, default True
        a boolean value that when set to True, this module has learnable per-element
        affine parameters initialized to ones (for weights) and zeros (for biases).
    Examples
    --------
    >>> input = torch.randn(20, 5, 10, 10)
    >>> # With Learnable Parameters
    >>> m = LazyLayerNorm()
    >>> # Without Learnable Parameters
    >>> m = LazyLayerNorm(elementwise_affine=False)
    >>> m
    LazyLayerNorm((0,), eps=1e-05, elementwise_affine=False)
    >>> output = m(input)
    >>> output.size()
    torch.Size([20, 5, 10, 10])
    >>> m
    LayerNorm((5, 10, 10), eps=1e-05, elementwise_affine=False)
    """

    cls_to_become = LayerNorm

    def __init__(self, eps=1e-5, elementwise_affine=True) -> None:
        super().__init__(0, eps, elementwise_affine)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def initialize_parameters(self, input) -> None:
        self.normalized_shape = tuple(input.size()[1:])
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(self.normalized_shape)
                self.bias.materialize(self.normalized_shape)
                
from torch_geometric.nn.conv import *

get_layer_impl_from_layer_type = {
        'GCNConv':{
            'impl':GCNConv,
            'params': {}
        },
        'ChebConv_symK1':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':1,'normalization':'sym'}
        }, 
        'ChebConv_symK2':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':2,'normalization':'sym'}
        }, 
        'ChebConv_symK3':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':3,'normalization':'sym'}
        }, 
        'ChebConv_rwK1':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':1,'normalization':'rw'}
        }, 
        'ChebConv_rwK2':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':2,'normalization':'rw'}
        }, 
        'ChebConv_rwK3':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':3,'normalization':'rw'}
        }, 
        'ChebConv_K1':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':1,'normalization':None}
        }, 
        'ChebConv_K2':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':2,'normalization':None}
        }, 
        'ChebConv_K3':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':3,'normalization':None}
        }, 
        'ChebConv_K4':{
            'impl':ChebConv,
            'params': {'default_convs': True, 'K':4,'normalization':None}
        }, 
        'SAGEConv':{
            'impl':SAGEConv,
            'params': {'normalize': True}
        },
        'SAGEConv_RootWeight':{
            'impl':SAGEConv,
            'params': {'default_convs': True, 'root_weight':True}
        },
        'GraphConv':{
            'impl':GraphConv,
            'params': {}
        },
        # 'GravNetConv':{ # "Static graphs not supported in 'GravNetConv
        #     'impl':GravNetConv,
        #     'params': {}
        # }, # added  space_dimensions: int= 10, propagate_dimensions: int=10, k: int=10, # num_layers: int = 1 #     return forward_call(*input, **kwargs)  "/home/dproios/miniconda3/envs/pyg_38_night/lib/python3.8/site-packages/torch_cluster/knn.py", line 55, in knn  assert x.size(0) == batch_x.numel()
        'GatedGraphConv':{
            'impl':GatedGraphConv,
            'params': {}
        },
        'ResGatedGraphConv':{
            'impl':ResGatedGraphConv,
            'params': {}
        },
        'GATConv':{
            'impl':GATConv,
            'params': {'heads': 4}
        },
        'GATv2Conv':{
            'impl':GATv2Conv,
            'params': {}
        },
        'TransformerConv':{
            'impl':TransformerConv,
            'params': {}
        },
        'AGNNConv':{
            'impl':AGNNConv,
            'params': {
                'requires_grad': True, 
                'add_self_loops':  True
            }
        }, # doesnt return 32 dim
        'TAGConv':{
            'impl':TAGConv,
            'params': {}
        },
        'GINConv':{
            'impl':GINConv,
            'params': {}
        },
        'GINEConv':{
            'impl':GINEConv,
            'params': {}
        }, # no edge feature :?
        'ARMAConv':{
            'impl':ARMAConv,
            'params': {}
        },
        'SGConv':{
            'impl':SGConv,
            'params': {}
        },
        'APPNP_k5t03':{
            'impl':APPNP,
            'params': {
                'K': 5, 
                'alpha': 0.3, 
                'dropout':  0.1,
                #cached: bool = False, add_self_loops: bool = True
            }
        },
        'APPNP_k1t05':{
            'impl':APPNP,
            'params': {
                'K': 1, 
                'alpha': 0.5, 
                'dropout':  0.1,
                #cached: bool = False, add_self_loops: bool = True
            }
        },'APPNP_k1t03':{
            'impl':APPNP,
            'params': {
                'K': 1, 
                'alpha': 0.3, 
                'dropout':  0.1,
                #cached: bool = False, add_self_loops: bool = True
            }
        },
        'APPNP_k5t05':{
            'impl':APPNP,
            'params': {
                'K': 5, 
                'alpha': 0.5, 
                'dropout':  0.1,
                #cached: bool = False, add_self_loops: bool = True
            }
        },
        'APPNP_k10t03':{
            'impl':APPNP,
            'params': {
                'K': 10, 
                'alpha': 0.3, 
                'dropout':  0.1,
                #cached: bool = False, add_self_loops: bool = True
            }
        },
        'APPNP_k10t05':{
            'impl':APPNP,
            'params': {
                'K': 10, 
                'alpha': 0.5, 
                'dropout':  0.1,
                #cached: bool = False, add_self_loops: bool = True
            }
        },
        'MFConv':{
            'impl':MFConv,
            'params': {}
        },
        'RGCNConv':{
            'impl':RGCNConv,
            'params': {}
        } ,
        'FastRGCNConv':{
            'impl':FastRGCNConv,
            'params': {}
        },
        #'RGATConv':{
        #    'impl':RGATConv,
        #    'params': {}
        #},
        'SignedConvFA':{
            'impl':SignedConv,
            'params': { 'default_convs': True,'first_aggr': True}
        },
        'SignedConvNFA':{
            'impl':SignedConv,
            'params': { 'default_convs': True,'first_aggr': False}
        },
        'DNAConv':{
            'impl':DNAConv,
            'params': { 'hidden_channels': True}
        },
        'DNAConv3h3g':{
            'impl':DNAConv,
            'params': { 'hidden_channels': True, 'heads': 3, 'groups': 3 }
        },
        # 'PointConv':{
        #     'impl':PointConv,
        #     'params': {}
        # },
        'GMMConv':{
            'impl':GMMConv,
            'params': {}
        },
        'SplineConv':{
            'impl':SplineConv,
            'params': {}
        },
        'NNConv':{
            'impl':NNConv,
            'params': {}
        },
        # 'ECConv':{
        #     'impl':ECConv,
        #     'params': {}
        # },
        'CGConv':{
            'impl':CGConv,
            'params': {}
        },
        'EdgeConv':{
            'impl':EdgeConv,
            'params': {}
        },
        'DynamicEdgeConv':{
            'impl':DynamicEdgeConv,
            'params': {}
        },
        'XConv':{
            'impl':XConv,
            'params': {}
        },
        'PPFConv':{
            'impl':PPFConv,
            'params': {}
        },
        'FeaStConv':{
            'impl':FeaStConv,
            'params': {}
        },
        'PointTransformerConv':{
            'impl':PointTransformerConv,
            'params': {}
        },
        'HypergraphConv':{
            'impl':HypergraphConv,
            'params': {}
        },
        'LEConv':{
            'impl':LEConv,
            'params': {}
        },
        'PNAConv':{
            'impl':PNAConv,
            'params': {}
        },
        'ClusterGCNConv':{
            'impl':ClusterGCNConv,
            'params': { "diag_lambda": 0.0 }
        },
        'GENConv':{
            'impl':GENConv,
            'params': {}
        },
        'GCN2Conv':{
            'impl':GCN2Conv,
            'params': {}
        },
        'PANConv':{
            'impl':PANConv,
            'params': {}
        },
        'WLConv':{
            'impl':WLConv,
            'params': {}
        },
        'FiLMConv':{
            'impl':FiLMConv,
            'params': {}
        },
        'SuperGATConv':{
            'impl':SuperGATConv,
            'params': {}
        }, # takes too long'
        'FAConv':{
            'impl':FAConv,
            'params': {}
        },
        'EGConv':{
            'impl':EGConv,
            'params': {}
        },
        'PDNConv':{
            'impl':PDNConv,
            'params': {}
        },
        'GeneralConv':{
            'impl':GeneralConv,
            'params': {}
        },
        'HGTConv':{
            'impl':HGTConv,
            'params': {}
        },
        'HEATConv':{
            'impl':HEATConv,
            'params': {}
        },
        # 'HeteroConv':{ # heterogenou
        #     'impl':HeteroConv,
        #     'params': {}
        # },
        # 'HANConv':{ # heterogenou
        #     'impl':HANConv,
        #     'params': {}
        # },
        #'LGConv':{
        #    'impl':LGConv,
        #    'params': { 'empty_init': True }
        #},
}

activation_name_implementation = {
    # 'relu': nn.ReLU,
    # 'leaky_relu': nn.LeakyReLU,
    # 'sigmoid': nn.Sigmoid,
    # 'tanh': nn.Tanh,
    # 'elu': nn.ELU,
    'selu': nn.SELU,
    'celu': nn.CELU,
    'gelu': nn.GELU,
    # 'softmax': nn.Softmax,
    # 'log_softmax': nn.LogSoftmax,
    # 'prelu': nn.PReLU,
    # 'softplus': nn.Softplus,
    # 'softsign': nn.Softsign,
    # 'tanhshrink': nn.Tanhshrink,
    # 'softmin': nn.Softmin,
    # 'hardtanh': nn.Hardtanh,
    # 'hardshrink': nn.Hardshrink,
    # 'hardswish': nn.Hardswish,
    # 'softshrink': nn.Softshrink,
    # 'threshold': nn.Threshold,
}
