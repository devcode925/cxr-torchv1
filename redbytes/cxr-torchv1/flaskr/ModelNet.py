import os,sys
sys.path.insert(0,"..")
from glob import glob
import numpy as np

import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
import skimage, skimage.filters
import torchxrayvision as xrv

class CovidModelNet(torch.nn.Module):
    def __init__(self):
        super(CovidModelNet, self).__init__()
        self.model = xrv.models.DenseNet(weights="all")
        self.model.op_threshs = None
        self.theta_bias_geographic_extent = torch.from_numpy(np.asarray((0.8705248236656189, 3.4137437)))
        self.theta_bias_opacity = torch.from_numpy(np.asarray((0.5484423041343689, 2.5535977)))

    def forward(self, x):
        preds = self.model(x)
        preds = preds[0,xrv.datasets.default_pathologies.index("Lung Opacity")]
        geographic_extent = preds*self.theta_bias_geographic_extent[0]+self.theta_bias_geographic_extent[1]
        opacity = preds*self.theta_bias_opacity[0]+self.theta_bias_opacity[1]
        geographic_extent = torch.clamp(geographic_extent,0,8)
        opacity = torch.clamp(opacity,0,6)
        return {"geographic_extent":geographic_extent,"opacity":opacity}