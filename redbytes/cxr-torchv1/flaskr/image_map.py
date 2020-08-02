def full_frame(width=None, height=None):
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    
def heat_map():
   

    img = img.requires_grad_()
    outputs = model2(img)
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
    plt.savefig(cfg.saliency_map)