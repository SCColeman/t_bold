"""
Module for analysing BOLD timecourses

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy


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
    def __init__(self, data, coords, tr):
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

    def norm_to_baseline(self, baseline_window):
        baseline_index = np.arange(baseline_window[0] / self.TR,
                                   baseline_window[1] / self.TR)
        baseline = np.mean(self.tcourse[baseline_index.astype(int)])
        tcourse = self.tcourse - baseline
        self.tcourse = tcourse

    def percentage_change(self, baseline_window):
        baseline_index = np.arange(baseline_window[0] / self.TR,
                                   baseline_window[1] / self.TR)
        baseline = np.mean(self.tcourse[baseline_index.astype(int)])
        tcourse = (self.tcourse / baseline) * 100
        self.tcourse = tcourse
        self.axlabel = "Percentage Change \n from Baseline (%)"

    def epoch(self, events, window, plot=True):
        epoch_len = int((window[1] - window[0]) / self.TR)
        epochs = np.zeros((len(events), epoch_len))
        for event in range(len(events)):
            event_i = np.arange((events[event] + window[0]) / self.TR,
                                (events[event] + window[1]) / self.TR)
            epochs[event, :] = self.tcourse[event_i.astype(int)]
        epoched = np.mean(epochs, axis=0)
        if not plot:
            return epoched
        else:
            fig, ax = plt.subplots()
            ax.plot(np.arange(0, epoch_len * self.TR, self.TR), epoched)
            ax.set_xlabel('Time (s)', fontsize=16)
            ax.set_ylabel(self.axlabel, fontsize=16)
            ax.set_title(self.title + ' (Epoched)', fontsize=16, fontweight='bold')
            fig.set_figheight(4)
            fig.set_figwidth(7)
            plt.tight_layout()
            return epoched, ax

    def epoch_conditions(self, events, window, plot=True):
        epoched = []
        epoch_len = int((window[1] - window[0]) / self.TR)
        for condition in range(len(events)):
            events_cond = events[condition]
            epoched.append(self.epoch(events_cond, window, plot=False))
        if plot:
            fig, ax = plt.subplots()
            for condition in range(len(events)):
                ax.plot(np.arange(0, epoch_len * self.TR, self.TR), epoched[condition])
            ax.set_xlabel('Time (s)', fontsize=16)
            ax.set_ylabel(self.axlabel, fontsize=16)
            ax.set_title(self.title + ' (Epoched)', fontsize=16, fontweight='bold')
            fig.set_figheight(4)
            fig.set_figwidth(7)
            plt.tight_layout()
            return epoched, ax
        else:
            return epoched

    def tstat(self, on_events, off_events, on_window, off_window):
        epochs_on = self.epoch(on_events, on_window, plot=False)
        epochs_off = self.epoch(off_events, off_window, plot=False)
        p_on = np.mean(epochs_on.flatten())
        p_off = np.mean(epochs_off.flatten())
        n_on = np.std(epochs_on.flatten())
        n_off = np.std(epochs_off.flatten())
        t = (p_on - p_off) / (n_on + n_off)
        return t