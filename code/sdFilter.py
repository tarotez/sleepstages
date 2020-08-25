from __future__ import print_function
from os.path import splitext
from parameterSetup import ParameterSetup
import numpy as np
import json

class SDFilter(object):

    def __init__(self):
        params = ParameterSetup()
        pickledDir = params.pickledDir
        self.fileIDs_L = []
        try:
            sd_threshHandler = open(pickledDir + '/' + 'sd_thresh.json', 'r')
            d = json.load(sd_threshHandler)
            self.W_amount_mean = d["W_amount_mean"]
            self.W_amount_2sd = d["W_amount_2sd"]
            self.R_amount_mean = d["R_amount_mean"]
            self.R_amount_2sd = d["R_amount_2sd"]
            self.S_amount_mean = d["S_amount_mean"]
            self.S_amount_2sd = d["S_amount_2sd"]

            self.W_episode_num_mean = d["W_episode_num_mean"]
            self.W_episode_num_2sd = d["W_episode_num_mean"]
            self.R_episode_num_mean = d["R_episode_num_mean"]
            self.R_episode_num_2sd = d["R_episode_num_mean"]
            self.S_episode_num_mean = d["S_episode_num_mean"]
            self.S_episode_num_2sd = d["S_episode_num_mean"]

            self.W_duration_mean = d["W_duration_mean"]
            self.W_duration_2sd = d["W_duration_mean"]
            self.R_duration_mean = d["R_duration_mean"]
            self.R_duration_2sd = d["R_duration_mean"]
            self.S_duration_mean = d["S_duration_mean"]
            self.S_duration_2sd = d["S_duration_mean"]
        except EnvironmentError:
            self.W_amount_mean = 0
            self.W_amount_2sd = 0
            self.R_amount_mean = 0
            self.R_amount_2sd = 0
            self.S_amount_mean = 0
            self.S_amount_2sd = 0

            self.W_episode_num_mean = 0
            self.W_episode_num_2sd = 0
            self.R_episode_num_mean = 0
            self.R_episode_num_2sd = 0
            self.S_episode_num_mean = 0
            self.S_episode_num_2sd = 0

            self.W_duration_mean = 0
            self.W_duration_2sd = 0
            self.R_duration_mean = 0
            self.R_duration_2sd = 0
            self.S_duration_mean = 0
            self.S_duration_2sd = 0

    def isOutlier(self, signal):   #### to be implmented (2017.9.25)
        std = np.std(signal)
        ### if std >= maxAllowedSTD:
        ###     return True
        ### else:
        ###    return False
        return False

    def notOutlier(self, signal):
        return not isOutlier(self, signal)
