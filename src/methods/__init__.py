# Import all method implementations to register them
from .method import Method
from .method_factory import MethodFactory, register_method
from .mc_dropout import MCDropout
from .tta import TTA
#from .swag import Swag
from .ddu import DDU
from .laplace_approximation import LaplaceApproximation
from .ensemble import Ensemble
from .het_xl import HetXL
from .entropy import Entropy
from .swag import Swag
from .influence_function import InfluenceFunction
from .evidential_dl import EvidentialDeepLearning

__all__ = [
    'Method', 'MethodFactory', 'register_method',
    'DDU', 'Ensemble', 'Entropy' , 'HetXL', 
    'LaplaceApproximation', 'MCDropout', 
    'Swag', 'TTA'
    ] 

