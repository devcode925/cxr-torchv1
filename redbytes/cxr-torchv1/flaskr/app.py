import os
import sys
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms
import skimage
import skimage.filters
import torchxrayvision as xrv
import io
# import jsonify

from ModelNet import CovidModelNet
from flask import Flask
from flask import request
from flask import send_file

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello chuck"


@app.route("/", methods=["GET", "POST"])
def work():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        incomingImg = request.files['image']
        print(incomingImg.filename)
        # save the file to ./uploads
        sys.path.insert(0, "..")
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(basepath, "uploads", incomingImg.filename)
        print(img_path)
        incomingImg.save(img_path)

        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]

        # Add color channel
        img = img[None, :, :]
        # now transform it
        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )
        img = transform(img)
        covid_model = CovidModelNet()
    
        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)

            outputs = covid_model(img)
            geo_extent = "{:1.4}".format(outputs["geographic_extent"].cpu().numpy())
            opactiy_val = "{:1.4}".format(outputs["opacity"].cpu().numpy())
            print("geographic_extent (0-8):", geo_extent)
            print("opacity (0-6):", opactiy_val)

        img = img.requires_grad_()
        outputs = covid_model(img)
        grads = torch.autograd.grad(outputs["geographic_extent"], img)[0][0][0]
        blurred = skimage.filters.gaussian(grads ** 2, sigma=(5, 5), truncate=3.5)

        full_frame()
        my_dpi = 100
        fig = plt.figure(
            frameon=False, figsize=(224 / my_dpi, 224 / my_dpi), dpi=my_dpi
        )
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img[0][0].detach(), cmap="gist_gray", aspect="auto")
        ax.imshow(blurred, alpha=0.5)
        heatmap = "heat_" + incomingImg.filename + ".png"
        print(heatmap)
        plt.savefig(heatmap)
        plt.close(fig)
        # need to fix mimetype to be dynamic based on file type.
        return send_file(heatmap, as_attachment=True, attachment_filename='heatmap', mimetype='image/png')

    return None


def full_frame(width=None, height=None):
    import matplotlib as mpl

    mpl.rcParams["savefig.pad_inches"] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
