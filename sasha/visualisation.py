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

    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(4, 10))
    fig.suptitle('Radiological Orientation', fontsize=16)
    ax1.imshow(np.rot90(slices[0], axes=(0, 1)), cmap='gray')
    ax1.scatter(voxel[1], img.shape[2] - 1 - voxel[2], c='red')
    ax2.imshow(np.rot90(slices[1], axes=(0, 1)), cmap='gray')
    ax2.scatter(voxel[0], img.shape[2] - 1 - voxel[2], c='red')
    ax3.imshow(np.rot90(slices[2], axes=(0, 1)), cmap='gray')
    ax3.scatter(voxel[0], img.shape[1] - 1 - voxel[1], c='red')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    plt.tight_layout()