# +
import os,sys
sys.path.insert(0,"..")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage
import pprint

import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
import skimage, skimage.filters
import torchxrayvision as xrv
from ModelNet import CovidModelNet

from flask import Flask
app = Flask(__name__)


# -

@app.route('/')
def hello():
    return 'Hello chuck'

# +
@app.route('/predict', methods=['GET', 'POST'])
    def work():
        if request.method == 'POST':
        # get the file from the HTTP-POST request
            incomingImg = request.files['image']        
            heatmap = heat + incomingImg.filename      
            print(heatmap)
            # save the file to ./uploads
            basepath = os.path.dirname(__file__)
            img_path = os.path.join(basepath, 'uploads', incomingImg.filename)
            f.save(img_path)
            
            img = skimage.io.imread(img_path)  
            img = xrv.datasets.normalize(img, 255)  
            # Add color channel
            img = img[None, :, :]   
            # now transform it
            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])
            img = transform(img)
            
            covid_model = CovidModelNet()
            img = img.requires_grad_()
            outputs = covid_model(img)
            grads = torch.autograd.grad(outputs["geographic_extent"], img)[0][0][0]
            blurred = skimage.filters.gaussian(grads**2, sigma=(5, 5), truncate=3.5)
    
            full_frame()
            my_dpi = 100
            fig = plt.figure(frameon=False, figsize=(224/my_dpi, 224/my_dpi), dpi=my_dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(img[0][0].detach(), cmap="gist_gray", aspect='auto')
            ax.imshow(blurred, alpha=0.5);
            plt.savefig(heatmap)
                              
            return result
                                                                                                
    return None

def full_frame(width=None, height=None):
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)

