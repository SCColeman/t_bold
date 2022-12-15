"""
Module to make higher level objects that combine runs/subjects

"""

from .timecourse_tools import *
from .registration import *
import nibabel as nib
import os
from dipy.io.image import load_nifti
from pathlib import Path
from scipy import stats

class HigherTimecourse:
    def __init__(self, data_fname_list, coords_list, TR, smooth=False):
        self.data_list = data_fname_list
        self.coords_list = coords_list
        self.TR = TR
        tcourse_list = []
        for lower in range(len(self.data_list)):
            data, affine = load_nifti(data_fname_list[lower])
            coords = self.coords_list[lower]
            tcourse_list.append(Timecourse(data, coords, self.TR, smooth))
        self.tcourse_list = tcourse_list
        self.axlabel = 'Amplitude (A.U)'

    def get_values(self):
        values = self.tcourse_list
        return values

    # pre-processing steps
    def percentage_change(self, baseline_window):
        for lower in range(len(self.tcourse_list)):
            self.tcourse_list[lower].percentage_change(baseline_window=baseline_window)
        self.axlabel = 'Percentage Change (%)'

    def remove_drift(self, block_len):
        for lower in range(len(self.tcourse_list)):
            self.tcourse_list[lower].remove_drift(block_len=block_len)

    def norm_to_baseline(self, baseline_window):
        for lower in range(len(self.tcourse_list)):
            self.tcourse_list[lower].norm_to_baseline(baseline_window=baseline_window)

    def separate_conditions(self, events_list, window):
        n_conds = len(events_list)
        n_runs = len(events_list[0])
        all_epochs = []
        for cond in range(n_conds):
            epochs_cond = []
            for run in range(n_runs):
                events_sublist = events_list[cond][run]
                lower_timecourse = self.tcourse_list[run].get_values()
                for event in events_sublist:
                    window_index = np.arange((event + window[0]) / self.TR, (event + window[1]) / self.TR)
                    epoch_i = lower_timecourse[window_index.astype(int)]
                    epochs_cond.append(epoch_i)
            all_epochs.append(epochs_cond)
        return all_epochs

    ### epoching
    def epoch(self, events, window):
        epoched = []
        for lower in range(len(self.tcourse_list)):
            epoched.append(epoch(self.tcourse_list[lower].get_values(), self.TR, events[lower], window, plot=False))
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
            ax = []
            return epoched_cond, ax

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
            ax = []
            return epoched_norm, ax


class HigherTimecourseFromArrays(HigherTimecourse):

    def __init__(self, data_fname_list, TR):
        self.TR = TR
        self.axlabel = 'Amplitude (A.U)'
        tcourse_list = []
        for lower in range(len(data_fname_list)):
            tcourse_list.append(Timecourse(data_fname_list[lower], coords=None,
                                           tr=self.TR))
        self.tcourse_list = tcourse_list



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


def time_window_t_test(timecourse_list, tr, window):
    n_subs = len(timecourse_list)
    n_conds = len(timecourse_list[0])
    window_index = np.arange(window[0] / tr, window[1] / tr)

    power_list = []
    for cond in range(n_conds):
        power_list_cond = []
        for sub in range(n_subs):
            timecourse = timecourse_list[sub][cond]
            power_list_cond.append(np.mean(timecourse[window_index.astype(int)]))
        power_list.append(power_list_cond)

    # shapiro test for normality
    for cond in range(n_conds):
        norm_test = stats.shapiro(power_list[cond])
        if norm_test[1] < 0.05:
            print('Data is likely not normally distributed,'
                  ' \nconsider different statistical test.')

    print(stats.ttest_rel(power_list[0], power_list[1]))


def plot_group_timecourse(timecourse_list, tr):
    n_subs = len(timecourse_list)
    n_conds = len(timecourse_list[0])
    n_points = len(timecourse_list[0][0])
    time = np.arange(0, n_points * tr, tr)

    timecourses = []
    errors = []
    for cond in range(n_conds):
        cond_timecourses = []
        for sub in range(n_subs):
            timecourse_sub = timecourse_list[sub][cond]
            cond_timecourses.append(timecourse_sub)
        timecourses.append(np.mean(cond_timecourses, axis=0))
        errors.append(np.std(cond_timecourses, axis=0))

    fig, ax = plt.subplots()
    plt.errorbar(time, timecourses[0], errors[0], alpha=0.7)
    plt.errorbar(time, timecourses[1], errors[1], alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('Group BOLD Timecourse (A.U)', fontsize=16)
    fig.set_figheight(4)
    fig.set_figwidth(7)
    plt.tight_layout()


def save_group_timecourse(timecourse_list, fname):
    n_subs = len(timecourse_list)
    n_conds = len(timecourse_list[0])
    n_points = len(timecourse_list[0][0])

    timecourses = []
    for cond in range(n_conds):
        cond_timecourses = []
        for sub in range(n_subs):
            timecourse_sub = timecourse_list[sub][cond]
            cond_timecourses.append(timecourse_sub)
        timecourses.append(np.mean(cond_timecourses, axis=0))

    savedata = np.array([timecourses[x] for x in range(n_conds)]).transpose()
    np.savetxt(fname, savedata)



