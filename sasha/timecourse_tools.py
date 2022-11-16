"""
Module for analysing BOLD timecourses

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype="high",
                               analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


class Timecourse:
    """
    Attributes
    data : np.array
        Time-series haemodynamic data at a single location.
    TR : int
        Repetition time of the fMRI sequence, i.e., sampling frequency in seconds.

    """
    def __init__(self, data, coords, tr, savepath=None, basename=None, specifier=None):
        self.path = savepath
        self.basename = basename
        self.data = data
        self.coords = coords
        self.TR = tr
        voxel = coords
        tcourse = data[voxel[0], voxel[1], voxel[2], :]
        self.tcourse = tcourse
        time = np.linspace(tr, len(tcourse) * tr, len(tcourse))
        self.time = time
        self.axlabel = 'Amplitude (A.U)'
        self.title = 'BOLD Data'
        if specifier is not None:
            self.timecourse_name = basename + '_' + specifier
        else:
            self.timecourse_name = basename

        if savepath is not None:
            if basename is None:
                raise Exception("If savepath is specified, you must supply a basename.")
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
            if not os.path.isdir(os.path.join(savepath, basename)):
                os.mkdir(os.path.join(savepath, basename))
            np.savetxt(os.path.join(savepath, basename, self.timecourse_name + '.txt'),
                       self.tcourse)

    def savefile(self):
        if self.path is not None:
            if self.basename is not None:
                np.savetxt(os.path.join(self.path, self.basename, self.timecourse_name + '.txt'),
                           self.tcourse)

    def get_values(self):
        values = self.tcourse
        return values

    def plot_tcourse(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.tcourse)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel(self.axlabel, fontsize=16)
        ax.set_title(self.title, fontsize=16, fontweight='bold')
        fig.set_figheight(4)
        fig.set_figwidth(7)
        plt.tight_layout()
        return ax

    def remove_drift(self, block_len):
        y = butter_highpass_filter(self.tcourse, 1 / (2 * block_len), 1 / self.TR)
        self.tcourse = y
        self.title = 'Highpass Filtered BOLD Data'
        if self.path is not None:
            self.timecourse_name = self.timecourse_name + '_hpass'
        self.savefile()

    def norm_to_baseline(self, baseline_window):
        baseline_index = np.arange(baseline_window[0] / self.TR,
                                   baseline_window[1] / self.TR)
        baseline = np.mean(self.tcourse[baseline_index.astype(int)])
        tcourse = self.tcourse - baseline
        self.tcourse = tcourse
        if self.path is not None:
            self.timecourse_name = self.timecourse_name + '_blined'
        self.savefile()

    def percentage_change(self, baseline_window):
        baseline_index = np.arange(baseline_window[0] / self.TR,
                                   baseline_window[1] / self.TR)
        baseline = np.mean(self.tcourse[baseline_index.astype(int)])
        tcourse = (self.tcourse / baseline) * 100
        self.tcourse = tcourse
        self.axlabel = "Percentage Change \n from Baseline (%)"
        if self.path is not None:
            self.timecourse_name = self.timecourse_name + '_pchange'
        self.savefile()


def epoch(data, tr, events, window, plot=True):
    epoch_len = int((window[1] - window[0]) / tr)
    epochs = np.zeros((len(events), epoch_len))
    for event in range(len(events)):
        event_i = np.arange((events[event] + window[0]) / tr,
                            (events[event] + window[1]) / tr)
        epochs[event, :] = data[event_i.astype(int)]
    epoched = np.mean(epochs, axis=0)
    if not plot:
        return epoched
    else:
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, epoch_len * tr, tr), epoched)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Epoched BOLD Data (A.U)', fontsize=16)
        fig.set_figheight(4)
        fig.set_figwidth(7)
        plt.tight_layout()
        return epoched, ax


def epoch_conditions(data, tr, events, window, plot=True):
    epoched = []
    epoch_len = int((window[1] - window[0]) / tr)
    for condition in range(len(events)):
        events_cond = events[condition]
        epoched.append(epoch(data, tr, events_cond, window, plot=False))
    if plot:
        fig, ax = plt.subplots()
        for condition in range(len(events)):
            ax.plot(np.arange(0, epoch_len * tr, tr), epoched[condition])
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Epoched BOLD Data (A.U)', fontsize=16)
        fig.set_figheight(4)
        fig.set_figwidth(7)
        plt.tight_layout()
        return epoched, ax
    else:
        return epoched


def tstat(data, tr, on_events, off_events, on_window, off_window):
    epochs_on = epoch(data, tr, on_events, on_window, plot=False)
    epochs_off = epoch(data, tr, off_events, off_window, plot=False)
    p_on = np.mean(epochs_on.flatten())
    p_off = np.mean(epochs_off.flatten())
    n_on = np.std(epochs_on.flatten())
    n_off = np.std(epochs_off.flatten())
    t = (p_on - p_off) / (n_on + n_off)
    return t


def find_peak_voxel(stat_map, mask):
    masked_stat = np.multiply(stat_map, mask)
    peak_voxel = np.where(masked_stat == np.amax(masked_stat))
    return peak_voxel
