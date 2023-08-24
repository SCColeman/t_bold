"""
Visualisation Module

"""

import numpy as np
import matplotlib.pyplot as plt


def image_voxel_slices(img, coords3d, t=0):
    voxel = coords3d
    slices = list()
    slices.append(img[voxel[0], :, :, t])
    slices.append(img[:, voxel[1], :, t])
    slices.append(img[:, :, voxel[2], t])

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(8, 4),
                                        facecolor='black')
    fig.suptitle('Radiological Orientation', fontsize=16, color='white')
    ax1.imshow(np.rot90(slices[0], axes=(0, 1)), cmap='gray')
    ax1.scatter(voxel[1], img.shape[2] - 1 - voxel[2], c='red')
    ax2.imshow(np.rot90(slices[1], axes=(0, 1)), cmap='gray')
    ax2.scatter(voxel[0], img.shape[2] - 1 - voxel[2], c='red')
    ax3.imshow(np.rot90(slices[2], axes=(0, 1)), cmap='gray')
    ax3.scatter(voxel[0], img.shape[1] - 1 - voxel[1], c='red')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')


def interactive_brain(img):
    init_coords = [50, 50, 25]
    if len(np.shape(img))==4:
        img = img[:, :, :, 0]
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(8, 4),
                                        facecolor='black')
    fig.suptitle('Radiological Orientation', fontsize=16, color='white')
    ax1.imshow(np.rot90(img[init_coords[0], :, :], axes=(0, 1)), cmap='gray')
    ax2.imshow(np.rot90(img[:, init_coords[1],:], axes=(0, 1)), cmap='gray')
    ax3.imshow(np.rot90(img[:, :, init_coords[2]], axes=(0, 1)), cmap='gray')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    # Make a horizontal slider to control the frequency.
    ax_xslider = fig.add_axes([0.1, 0.1, 0.2, 0.03])
    x_slider = plt.Slider(
        ax=ax_xslider,
        label='x coord',
        valmin=0,
        valmax=np.shape(img)[0]-1,
        valinit=init_coords[0],
        valstep=1
    )

    # Make a horizontal slider to control the frequency.
    ax_yslider = fig.add_axes([0.4, 0.1, 0.2, 0.03])
    y_slider = plt.Slider(
        ax=ax_yslider,
        label='y coord',
        valmin=0,
        valmax=np.shape(img)[1] - 1,
        valinit=init_coords[1],
        valstep=1
    )

    # Make a horizontal slider to control the frequency.
    ax_zslider = fig.add_axes([0.7, 0.1, 0.2, 0.03])
    z_slider = plt.Slider(
        ax=ax_zslider,
        label='z coord',
        valmin=0,
        valmax=np.shape(img)[2] - 1,
        valinit=init_coords[2],
        valstep=1
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        ax1.imshow(np.rot90(img[x_slider.val, :, :], axes=(0, 1)), cmap='gray')
        ax2.imshow(np.rot90(img[:, y_slider.val, :], axes=(0, 1)), cmap='gray')
        ax3.imshow(np.rot90(img[:, :, z_slider.val], axes=(0, 1)), cmap='gray')

    # register the update function with each slider
    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)


def overlay_statmap(img, statmap, thresh):
    condition = np.logical_and(statmap < np.percentile(statmap, thresh), statmap > np.percentile(statmap, 100-thresh))
    statmap_masked = np.ma.masked_where(condition, statmap)
    cmap = 'bwr'
    init_coords = [50, 50, 25]
    if len(np.shape(img))==4:
        img = img[:, :, :, 0]
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(8, 4),
                                        facecolor='black')
    fig.suptitle('Radiological Orientation', fontsize=16, color='white')
    ax1.imshow(np.rot90(img[init_coords[0], :, :], axes=(0, 1)), cmap='gray')
    ax2.imshow(np.rot90(img[:, init_coords[1],:], axes=(0, 1)), cmap='gray')
    ax3.imshow(np.rot90(img[:, :, init_coords[2]], axes=(0, 1)), cmap='gray')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax1o = fig.add_axes(ax1.get_position())
    ax2o = fig.add_axes(ax2.get_position())
    ax3o = fig.add_axes(ax3.get_position())
    ax1o.imshow(np.rot90(statmap_masked[init_coords[0], :, :], axes=(0, 1)), cmap=cmap)
    ax2o.imshow(np.rot90(statmap_masked[:, init_coords[1], :], axes=(0, 1)), cmap=cmap)
    ax3o.imshow(np.rot90(statmap_masked[:, :, init_coords[2]], axes=(0, 1)), cmap=cmap)
    ax1o.axis('off')
    ax2o.axis('off')
    ax3o.axis('off')

    # Make a horizontal slider to control the frequency.
    ax_xslider = fig.add_axes([0.1, 0.1, 0.2, 0.03])
    x_slider = plt.Slider(
        ax=ax_xslider,
        label='x coord',
        valmin=0,
        valmax=np.shape(img)[0]-1,
        valinit=init_coords[0],
        valstep=1
    )

    # Make a horizontal slider to control the frequency.
    ax_yslider = fig.add_axes([0.4, 0.1, 0.2, 0.03])
    y_slider = plt.Slider(
        ax=ax_yslider,
        label='y coord',
        valmin=0,
        valmax=np.shape(img)[1] - 1,
        valinit=init_coords[1],
        valstep=1
    )

    # Make a horizontal slider to control the frequency.
    ax_zslider = fig.add_axes([0.7, 0.1, 0.2, 0.03])
    z_slider = plt.Slider(
        ax=ax_zslider,
        label='z coord',
        valmin=0,
        valmax=np.shape(img)[2] - 1,
        valinit=init_coords[2],
        valstep=1
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        ax1.clear()
        ax1.imshow(np.rot90(img[x_slider.val, :, :], axes=(0, 1)), cmap='gray')
        ax1o.clear()
        ax1o.imshow(np.rot90(statmap_masked[x_slider.val, :, :], axes=(0, 1)), cmap=cmap)
        ax1o.axis('off')
        ax2.imshow(np.rot90(img[:, y_slider.val, :], axes=(0, 1)), cmap='gray')
        ax2o.clear()
        ax2o.imshow(np.rot90(statmap_masked[:, y_slider.val, :], axes=(0, 1)), cmap=cmap)
        ax2o.axis('off')
        ax3.imshow(np.rot90(img[:, :, z_slider.val], axes=(0, 1)), cmap='gray')
        ax3o.clear()
        ax3o.imshow(np.rot90(statmap_masked[:, :, z_slider.val], axes=(0, 1)), cmap=cmap)
        ax3o.axis('off')

    # register the update function with each slider
    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)
