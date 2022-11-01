"""
Module to make higher level objects that combine runs/subjects

"""

from .timecourse_tools import *


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