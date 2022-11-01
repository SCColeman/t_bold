"""
Module for applying timecourse functionality to whole brain.

"""

from .timecourse_tools import *


class Brain:
    def __init__(self, img, TR):
        self.data = img
        self.TR = TR
        self.dims = np.shape(self.data)

    def get_data(self):
        return self.data

    def remove_drift(self, block_len):
        brain_filtered = np.zeros(self.dims)
        b, a = butter_highpass(1 / (2 * block_len), 1 / self.TR)
        for row in np.arange(self.dims[0]):
            for col in np.arange(self.dims[1]):
                for sli in np.arange(self.dims[2]):
                    timecourse = self.data[row, col, sli, :]
                    y = butter_highpass_filter(timecourse, 0.005, 1 / self.TR)
                    brain_filtered[row, col, sli, :] = y
        self.data = brain_filtered

    def percentage_change(self, baseline_window):
        brain_preprocced = np.zeros(self.dims)
        for row in np.arange(self.dims[0]):
            for col in np.arange(self.dims[1]):
                for sli in np.arange(self.dims[2]):
                    timcourse = Timecourse(self.data, [row, col, sli], self.TR)
                    timcourse.percentage_change(baseline_window=baseline_window)
                    brain_preprocced[row, col, sli, :] = timcourse.get_values()
        self.data = brain_preprocced

    def norm_to_baseline(self, baseline_window):
        brain_preprocced = np.zeros(self.dims)
        for row in np.arange(self.dims[0]):
            for col in np.arange(self.dims[1]):
                for sli in np.arange(self.dims[2]):
                    timecourse = Timecourse(self.data, [row, col, sli], self.TR)
                    timecourse.norm_to_baseline(baseline_window=baseline_window)
                    brain_preprocced[row, col, sli, :] = timecourse.get_values()
        self.data = brain_preprocced

    def brain_tstat(self, on_events, off_events, on_window, off_window):
        t = np.zeros(self.dims[0:3])
        for row in np.arange(self.dims[0]):
            for col in np.arange(self.dims[1]):
                for sli in np.arange(self.dims[2]):
                    voxel_timecourse = Timecourse(self.data, [row, col, sli], self.TR)
                    t[row, col, sli] = voxel_timecourse.tstat(on_events, off_events,
                                                              on_window, off_window)
        return t