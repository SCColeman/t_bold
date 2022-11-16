"""
Module to make higher level objects that combine runs/subjects

"""

from .timecourse_tools import *
from .registration import *
import nibabel as nib
import os
from dipy.io.image import load_nifti
from pathlib import Path

class HigherTimecourse:
    def __init__(self, data_list, coords_list, TR):
        self.data_list = data_list
        self.coords_list = coords_list
        self.TR = TR
        tcourse_list = []
        for lower in range(len(self.data_list)):
            data = self.data_list[lower]
            coords = self.coords_list[lower]
            tcourse_list.append(Timecourse(data, coords, self.TR))
        self.tcourse_list = tcourse_list
        self.axlabel = 'Amplitude (A.U)'

    def get_values(self):
        values = self.tcourse_list
        return values

    # pre-processing steps
    def average_inputs(self):
        mean_tcourse = np.mean([self.tcourse_list[x].get_values() for x in range(len(self.data_list))], axis=0)
        return mean_tcourse

    def percentage_change(self, baseline_window):
        for lower in range(len(self.data_list)):
            self.tcourse_list[lower].percentage_change(baseline_window=baseline_window)
        self.axlabel = 'Percentage Change (%)'

    def remove_drift(self, block_len):
        for lower in range(len(self.data_list)):
            self.tcourse_list[lower].remove_drift(block_len=block_len)

    def norm_to_baseline(self, baseline_window):
        for lower in range(len(self.data_list)):
            self.tcourse_list[lower].norm_to_baseline(baseline_window=baseline_window)

    ### epoching
    def epoch(self, events, window):
        epoched = []
        for lower in range(len(self.data_list)):
            epoched.append(self.tcourse_list[lower].epoch(events[lower], window, plot=False))
        return epoched

    def epoch_conditions(self, events, window, plot=True):
        epoch_len = int((window[1] - window[0]) / self.TR)
        epoched_cond = []
        for condition in range(len(events)):
            events_cond = events[condition]
            epoched_lowers = self.epoch(events_cond, window)
            epoched_cond.append(np.mean(epoched_lowers, axis=0))
        if plot:
            fig, ax = plt.subplots()
            for condition in range(len(events)):
                ax.plot(np.arange(0, epoch_len * self.TR, self.TR), epoched_cond[condition])
            ax.set_xlabel('Time (s)', fontsize=16)
            ax.set_ylabel(self.axlabel, fontsize=16)
            fig.set_figheight(4)
            fig.set_figwidth(7)
            plt.tight_layout()
            return epoched_cond, ax
        else:
            return epoched_cond

    def epoch_norm_to_window(self, epoched, window, plot=True):
        epoched_norm = []
        epoch_len = len(epoched[0])
        window_i = np.arange(window[0] / self.TR, window[1] / self.TR)
        for condition in range(len(epoched)):
            epoched_norm.append(epoched[condition] /
                                np.mean(epoched[condition][window_i.astype(int)]))
        if plot:
            fig, ax = plt.subplots()
            for condition in range(len(epoched)):
                ax.plot(np.arange(0, epoch_len * self.TR, self.TR), epoched_norm[condition])
            ax.set_xlabel('Time (s)', fontsize=16)
            ax.set_ylabel('Relative Change (A.U)', fontsize=16)
            fig.set_figheight(4)
            fig.set_figwidth(7)
            plt.tight_layout()
            return epoched_norm, ax
        else:
            return epoched_norm


def higher_level_statmap(statmap_fname_list, standard_fname, functional_fname, anatomical_fname,
                         savepath, basename, identifier):
    statmap_stand_list = register_func2standard(statmap_fname_list, standard_fname,
                                                functional_fname, anatomical_fname, True)
    standard, standard_affine = load_nifti(standard_fname)

    # save individual t-stats in standard space
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    if not os.path.isdir(os.path.join(savepath, basename)):
        os.mkdir(os.path.join(savepath, basename))

    for i in range(len(statmap_fname_list)):
        statmap_fname = os.path.basename(statmap_fname_list[i])
        if os.path.splitext(statmap_fname)[1] == '.gz':
            output_fname = statmap_fname[:-7] + '_standard.nii.gz'
        elif os.path.splitext(statmap_fname)[1] == '.nii':
            output_fname = statmap_fname[:-4] + '_standard.nii.gz'
        nib.save(nib.Nifti1Image(statmap_stand_list[i], standard_affine),
                os.path.join(savepath, basename, output_fname))

    # combine t-stat maps
    statmap_avg = np.mean(statmap_stand_list, axis=0)
    output_fname = basename + '_' + identifier + '.nii.gz'
    nib.save(nib.Nifti1Image(statmap_avg, standard_affine),
             os.path.join(savepath, basename, output_fname))

    return statmap_avg


class HigherStat:
    def __init__(self, savepath, higher_basename, lowlevel_basenames):
        self.savepath = savepath
        self.basename = higher_basename
        self.lowlevel = lowlevel_basenames
        self.func2anat = None
        self.anat2stand = None
        self.standard = None
        self.standard_affine = None
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        if not os.path.isdir(os.path.join(savepath, higher_basename)):
            os.mkdir(os.path.join(savepath, higher_basename))

    def calculate_affine(self, standard_fname, functional_fname,
                         anatomical_fname, plot=True):
        self.standard, self.standard_affine = load_nifti(standard_fname)
        func_standspace, affine_func2anat, affine_anat2stand = \
            register_func2standard(functional_fname, standard_fname,
                                   functional_fname, anatomical_fname,
                                   plot)
        self.func2anat = affine_func2anat
        self.anat2stand = affine_anat2stand
        output_fname = self.basename + '_func_standspace.nii.gz'
        nib.save(nib.Nifti1Image(func_standspace, self.standard_affine),
                 os.path.join(self.savepath, self.basename, output_fname))

    def statmap2standard(self, identifier):
        all_statmaps = []
        # transform all low level stat maps to standard space
        for lowlevel in self.lowlevel:
            statmap_fname = os.path.join(self.savepath, lowlevel,
                                         lowlevel + '_' + identifier +
                                         '.nii.gz')
            statmap, statmap_affine = load_nifti(statmap_fname)
            statmap_anatspace = self.func2anat.transform(statmap)
            statmap_standspace = self.anat2stand.transform(statmap_anatspace)
            all_statmaps.append(statmap_standspace)
            output_fname = lowlevel + '_' + identifier + '_standard.nii.gz'
            nib.save(nib.Nifti1Image(statmap_standspace, self.standard_affine),
                     os.path.join(self.savepath, self.basename, output_fname))

        # take average of stat maps
        avg_statmap = np.mean(all_statmaps, axis=0)
        output_fname = self.basename + '_' + identifier + '_standard.nii.gz'
        nib.save(nib.Nifti1Image(avg_statmap, self.standard_affine),
                 os.path.join(self.savepath, self.basename, output_fname))
        return avg_statmap

