"""
Module for applying timecourse functionality to whole brain.

"""

from .timecourse_tools import *
import nibabel as nib
from scipy.ndimage import gaussian_filter
import numpy as np

class Brain:
    def __init__(self, img_path, tr, savepath=None, basename=None, specifier=None):
        img = nib.load(img_path)
        self.data = img.get_fdata()
        self.affine = img.affine
        self.path = savepath
        self.basename = basename
        self.tr = tr
        self.dims = np.shape(self.data)
        self.pixdims = img.header.get_zooms()
        if specifier is not None:
            self.data_name = basename + '_' + specifier
        else:
            self.data_name = basename

        if savepath is not None:
            if basename is None:
                raise Exception("If savepath is specified, you must supply a basename.")
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
            if not os.path.isdir(os.path.join(savepath, basename)):
                os.mkdir(os.path.join(savepath, basename))
            nib.save(nib.Nifti1Image(self.data, self.affine),
                     os.path.join(savepath, basename, self.data_name + '.nii.gz'))

    def savefile(self):
        if self.path is not None:
            if self.basename is not None:
                nib.save(nib.Nifti1Image(self.data, self.affine),
                         os.path.join(self.path, self.basename, self.data_name + '.nii.gz'))

    def get_data(self):
        return self.data

    def gauss_smooth(self, fwhm, save=False):
        fwhm_pix = np.array(fwhm / np.array(self.pixdims))
        print(self.pixdims)
        print(fwhm_pix)
        sigma = fwhm_pix[0:3] / 2.355
        print(sigma)
        brain_smoothed = np.zeros(self.dims)
        for t in range(self.dims[3]):
            vol = np.squeeze(self.data[:, :, :, t])
            brain_smoothed[:, :, :, t] = gaussian_filter(vol, sigma)
        self.data = brain_smoothed
        self.data_name = self.data_name + '_smoothed'
        if save:
            self.savefile()


    def remove_drift(self, block_len, save=False):
        brain_filtered = np.zeros(self.dims)
        b, a = butter_highpass(1 / (2 * block_len), 1 / self.tr)
        for row in np.arange(self.dims[0]):
            for col in np.arange(self.dims[1]):
                for sli in np.arange(self.dims[2]):
                    timecourse = self.data[row, col, sli, :]
                    y = butter_highpass_filter(timecourse, 0.005, 1 / self.tr)
                    brain_filtered[row, col, sli, :] = y
        self.data = brain_filtered
        self.data_name = self.data_name + '_hpass'
        if save:
            self.savefile()

    def percentage_change(self, baseline_window, save=False):
        brain_preprocced = np.zeros(self.dims)
        for row in np.arange(self.dims[0]):
            for col in np.arange(self.dims[1]):
                for sli in np.arange(self.dims[2]):
                    timcourse = Timecourse(self.data, [row, col, sli], self.tr)
                    timcourse.percentage_change(baseline_window=baseline_window)
                    brain_preprocced[row, col, sli, :] = timcourse.get_values()
        self.data = brain_preprocced
        self.data_name = self.data_name + '_pchange'
        if save:
            self.savefile()

    def norm_to_baseline(self, baseline_window, save=False):
        brain_preprocced = np.zeros(self.dims)
        for row in np.arange(self.dims[0]):
            for col in np.arange(self.dims[1]):
                for sli in np.arange(self.dims[2]):
                    timecourse = Timecourse(self.data, [row, col, sli], self.tr)
                    timecourse.norm_to_baseline(baseline_window=baseline_window)
                    brain_preprocced[row, col, sli, :] = timecourse.get_values()
        self.data = brain_preprocced
        self.data_name = self.data_name + '_rezeroed'
        if save:
            self.savefile()

    def brain_tstat(self, on_events, off_events, on_window, off_window):
        t = np.zeros(self.dims[0:3])
        for row in np.arange(self.dims[0]):
            for col in np.arange(self.dims[1]):
                for sli in np.arange(self.dims[2]):
                    voxel_timecourse = Timecourse(self.data, [row, col, sli], self.tr)
                    t[row, col, sli] = tstat(voxel_timecourse.get_values(), self.tr, on_events, off_events,
                                                              on_window, off_window)
        return t

    def brain_connmap(self, events, window):
        connmap = np.zeros(self.dims[0:3])
        for row1 in np.arange(self.dims[0]):
            for col1 in np.arange(self.dims[1]):
                for sli1 in np.arange(self.dims[2]):
                    voxel_timecourse1 = Timecourse(self.data, [row1, col1, sli1], self.tr).get_values()
                    epoched_timecourse1 = epoch(voxel_timecourse1, self.tr, )
                    for row2 in np.arange(self.dims[0]):
                        for col2 in np.arange(self.dims[1]):
                            for sli2 in np.arange(self.dims[2]):
                                voxel_timecourse2 = Timecourse(self.data, [row2, col2, sli2], self.tr).get_values()
                                connmap[row2, col2, sli2] = np.corrcoef(voxel_timecourse1, voxel_timecourse2)
        return connmap


def save_img(data, brain_obj, identifier):
    ni_img = nib.Nifti1Image(data, brain_obj.affine)
    output_path = os.path.join(brain_obj.path, brain_obj.basename, brain_obj.basename +
                               '_' + identifier + '.nii.gz')
    nib.save(ni_img, output_path)

