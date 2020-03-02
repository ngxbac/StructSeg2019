# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from losses import *
from callbacks import *
from optimizers import *
from schedulers import *
from segmentation_models_pytorch import Unet as smpUnet
from segmentation_models_pytorch import FPN

import torchvision


# Register models
registry.Model(UNet3D)
registry.Model(UNet3D2)
registry.Model(ResidualUNet3D)
registry.Model(VNet)
registry.Model(smpUnet)
registry.Model(FPN)
# registry.Model(DeepLab)
registry.MODELS._late_add_callbacks = []

# Register callbacks
registry.Callback(MultiDiceCallback)

# Register criterions
registry.Criterion(MultiDiceLoss)

# Register optimizers
# registry.Optimizer(AdamW)
# registry.Optimizer(Nadam)

# registry.Scheduler(CyclicLRFix)