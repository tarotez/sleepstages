from __future__ import print_function
from os import mkdir
from os.path import isdir
import numpy as np
from stageLabelAndOneHot import stageLabel2oneHot
from inputAndOutputExtractor import InputAndOutputExtractor
import torch
import torch.utils.data
from torch import nn, optim, cat
from functools import reduce
from operator import mul
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torchsummary import summary

'''
# uses CNN for both raw signal and frequency histogram (spectrum)
class double_cnn_lstm(nn.Module):

    def conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=np.int(np.floor((kernel_size-1)/2)), bias=False)

    def __init__(self, params, num_classes):
        super(cnn_lstm, self).__init__()
        self.params = params
        self.input_channelNum = 1
        self.rawDataDim = params.downsample_outputDim
        self.lstm_subseqLen = params.torch_lstm_length
        self.lstm_num_layers = params.torch_lstm_num_layers
        self.lstm_hidden_size = params.torch_lstm_hidden_size
        self.lstm_inputDim = params.torch_lstm_inputDim
        self.lstm_bidirectional = True if params.torch_lstm_bidirectional == 1 else False
        if params.useFreqHisto and params.useTime:
            self.additionalFeatureDim = params.additionalFeatureDim
        elif params.useFreqHisto and not params.useTime:
            self.additionalFeatureDim = params.additionalFeatureDim - 1
        elif not params.useFreqHisto and params.useTime:
            self.additionalFeatureDim = 1
        else:
            self.additionalFeatureDim = 0
        if params.useRawData:
            self.input_shape = (self.lstm_subseqLen, self.input_channelNum, self.rawDataDim + self.additionalFeatureDim)
        else:
            self.input_shape = (self.lstm_subseqLen, self.input_channelNum, self.additionalFeatureDim)
        self.filter_nums = params.torch_filter_nums
        self.kernel_sizes = params.torch_kernel_sizes
        self.strides = params.torch_strides
        self.rawData_final_channel_num = self.filter_nums[-1]
        self.dropoutRate = params.dropoutRate
        self.skip_by = params.torch_skip_by
        self.layerNum = len(self.filter_nums)
        # self.avg_pool_size = params.torch_resnet_avg_pool_size
        # self.avg_pool = nn.AvgPool1d(self.avg_pool_size)
        # print('# in __init__, self.rawDataDim =', self.rawDataDim)
        # print('# in __init__, self.strides =', self.strides)

        # print('# in __init__, self.rawData_final_channel_num = ', self.rawData_final_channel_num, ', self.rawData_outputDim =', self.rawData_outputDim)
        # print('# in __init__, self.combined_size = ', self.combined_size)
        # self.batn_first = nn.BatchNorm1d(1)

        self.relu = nn.ReLU(inplace=True)
        in_channel_num = 1
        skip_in_channel_num = in_channel_num
        skip_inputDim = self.rawDataDim
        skip_in_blockID = 0
        batns = []
        convs = []
        drops = []
        skips = []
        for blockID, (out_channel_num, kernel_size, stride) in enumerate(zip(self.filter_nums, self.kernel_sizes, self.strides)):
            batns.append(nn.BatchNorm1d(in_channel_num))
            convs.append(self.conv(in_channel_num, out_channel_num, kernel_size, stride))
            drops.append(nn.Dropout(self.dropoutRate))
            in_channel_num = out_channel_num

            if blockID % self.skip_by == (self.skip_by - 1):
                # print('# blockID =', blockID)
                skip_out_layerID = blockID + 1
                skip_outputDim = reduce(lambda a, x: np.int(np.ceil((a - (x[0] - 1)) / x[1])), zip(self.kernel_sizes[skip_in_blockID:skip_out_layerID], self.strides[skip_in_blockID:skip_out_layerID]), skip_inputDim)
                skip_kernel_size = self.kernel_sizes[skip_in_blockID]
                skip_stride = reduce(lambda a, x: a * x, self.strides[skip_in_blockID:skip_out_layerID], 1)
                # print('# skip_inputDim =', skip_inputDim, ', skip_kernel_size =', skip_kernel_size)
                # print('# skip_outputDim =', skip_outputDim, ', skip_stride =', skip_stride)
                skip_out_channel_num = out_channel_num
                skips.append(self.conv(skip_in_channel_num, skip_out_channel_num, kernel_size=skip_kernel_size, stride=skip_stride))
                skip_in_channel_num = skip_out_channel_num
                skip_inputDim = skip_outputDim
                skip_in_blockID = skip_out_layerID
        self.batns = nn.ModuleList(batns)
        self.convs = nn.ModuleList(convs)
        self.drops = nn.ModuleList(drops)
        self.skips = nn.ModuleList(skips)

        # self.rawData_outputDim = reduce(lambda a, x: np.int(np.ceil((a - (x[0] - 1)) / x[1])), zip(self.kernel_sizes, self.strides), self.rawDataDim)
        self.rawData_outputDim = reduce(lambda a, x: np.int(np.ceil(a / x[1])), zip(self.kernel_sizes, self.strides), self.rawDataDim)
        if self.params.useRawData:
            self.combined_size = (self.rawData_final_channel_num * self.rawData_outputDim) + self.additionalFeatureDim
        else:
            self.combined_size = self.additionalFeatureDim

        # print('%%% in deepClassifier.py, combined_size =', self.combined_size)
        self.batn_combined = nn.BatchNorm1d(self.combined_size)
        self.fulc_combined = nn.Linear(self.combined_size, self.lstm_inputDim)
        self.dropout_combined = nn.Dropout(self.dropoutRate)
        self.lstm = nn.LSTM(input_size=self.lstm_inputDim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional, dropout=self.dropoutRate, batch_first=True)
        ### self.lstm = nn.GRU  (input_size=self.lstm_inputDim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional, dropout=self.dropoutRate, batch_first=True)
        self.num_of_directions = 2 if self.lstm_bidirectional else 1
        self.final_fc = nn.Linear(self.lstm_hidden_size * self.num_of_directions, num_classes)

    def forward(self, x):
        # print('$%$%$%$%$% in forward, x.shape = (batchSize, seqLen, channelNum, featureNum) = ', x.shape)
        # normalized = self.batn_first(x)
        # connect two parts
        batchSize, subseqLen, channelNum, featureNum = x.shape
        x = x.reshape(batchSize * subseqLen, channelNum, featureNum)
        # print('$%$%$%$%$% in forward, after reshape, x.shape = (new_batchSize, channelNum, featureNum) = ', x.shape)
        if not self.params.useRawData:
            additionalFeaturePart = x
            combined = additionalFeaturePart.view(additionalFeaturePart.size(0), -1)
        else:
            rawDataPart = x[:,:,:self.rawDataDim]
            rawDataCopy = rawDataPart
            skipOriginLayer = 0
            skipID = 0
            for blockID in range(len(self.convs)):
                rawDataPart = self.batns[blockID](rawDataPart)
                rawDataPart = self.convs[blockID](rawDataPart)
                rawDataPart = self.relu(rawDataPart)
                if self.dropoutRate > 0:
                    rawDataPart = self.drops[blockID](rawDataPart)

                if blockID % self.skip_by == (self.skip_by - 1):
                    rawDataCopy = self.skips[skipID](rawDataCopy)
                    rawDataPart = rawDataPart + rawDataCopy
                    rawDataCopy = rawDataPart
                    skipDestinLayer = blockID + 1
                    ### print('$%$%$%$%$%$%$%$ skip connection from layer ', skipOriginLayer, ' to layer ', skipDestinLayer, sep='')
                    skipOriginLayer = skipDestinLayer
                    skipID += 1
            # rawDataPart = self.avg_pool(rawDataPart)
            if self.additionalFeatureDim > 0:
                additionalFeaturePart = x[:,:,self.rawDataDim:]
                combined = cat((rawDataPart.view(rawDataPart.size(0), -1), additionalFeaturePart.view(additionalFeaturePart.size(0), -1)), dim=1)
            else:
                combined = rawDataPart.view(rawDataPart.size(0), -1)

        # print('%%% in forward(), combined.shape = ', combined.shape)
        combined = self.batn_combined(combined)
        if self.dropoutRate > 0:
            combined = self.dropout_combined(combined)
        # print('# before fulc, combined.shape =', combined.shape)
        combined = self.fulc_combined(combined)
        # print('# after fulc, combined.shape =', combined.shape)
        ######
        sequences = combined.reshape(batchSize, subseqLen, -1)
        ######
        # print('# in forward, sequences.shape =', sequences.shape)
        # print('# in forward, sequences =', sequences)
        lstm_output, _ = self.lstm(sequences)
        # print('# in forward, lstm_output.shape = (batchSize, subseqLen, lstm_outputDim) ', lstm_output.shape)
        last_element_of_lstm_output = lstm_output[:,-1,:]
        # print('# in forward, last_element_of_lstm_output.shape = (batchSize, lstm_outputDim) ', last_element_of_lstm_output.shape)
        final_output = self.final_fc(last_element_of_lstm_output)
        # print('# final_output.shape = (batchSize, classNum) = ', final_output.shape)
        return final_output
'''

# mean false error by Wang+16, IJCNN.
class MFE_Loss(torch.nn.Module):
    def __init__(self):
        super(MFE_Loss, self).__init__()
        print('using MFE_Loss')

    def set_orig_loss(self, orig_loss):
        self.orig_loss = orig_loss

    def forward(self, inputs, targets):
        # print('%$%$%$ inputs.shape =', inputs.shape)
        # print('%$%$%$ targets.shape =', targets.shape)
        labelTypes = set(targets)
        input_tensorDim = [0] + list(inputs.shape[1:])
        # print('input_tensorDim =', input_tensorDim)
        groupedInputs = [torch.empty(input_tensorDim, dtype=torch.float) for _ in labelTypes]
        groupedTargets = [torch.empty((0), dtype=torch.long) for _ in labelTypes]
        # print('groupedInputs[0].shape =', groupedInputs[0].shape)
        # print('groupedTargets[0].shaoe =', groupedTargets[0].shape)
        label2ID = {label.item() : ID for ID, label in enumerate(labelTypes)}
        for input, target in zip(inputs, targets):
            input_reshaped = input.view((1,-1))
            target_tensor = torch.as_tensor(np.array([target], dtype=np.long), dtype=torch.long)
            groupedInputs[label2ID[target.item()]] = torch.cat([groupedInputs[label2ID[target.item()]], input_reshaped])
            groupedTargets[label2ID[target.item()]] = torch.cat([groupedTargets[label2ID[target.item()]], target_tensor])
            # print('input =', input)
            # print('target = ', target)
            # print('label2ID =', label2ID)
            # print('target =', target.item())
            # print('type(groupedInputs[label2ID[target.item()]]) =', type(groupedInputs[label2ID[target.item()]]))
            # print('type(input) =', type(input))
            # print('groupedInputs[label2ID[target.item()]].shape =', groupedInputs[label2ID[target.item()]].shape)
            # print('input.shape =', input.shape)
            ### print('input_reshaped.shape =', input_reshaped.shape)
            # print('target.shape =', target.shape)
            ### print('target_tensor.shape =', target_tensor.shape)
            # if input_reshaped.shape[0] != target_tensor.shape[0]:
            #    print('$$$$$$$$$')
            #    print('input_reshaped.shape[0] != target_tensor.shape[0]')
            #    print('input_reshaped.shape =', input_reshaped.shape)
            #    print('target_tensor.shape =', target_tensor.shape)
            # print('groupedTargets[label2ID[target.item()]].shape =', groupedTargets[label2ID[target.item()]].shape)
            # print('groupedInputs[label2ID[target.item()]].shape =', groupedInputs[label2ID[target.item()]].shape)
            # print('groupedTargets[label2ID[target.item()]].shaoe =', groupedTargets[label2ID[target.item()]].shape)
            def g2t(grouped, label):
                grouped[label2ID[label.item()]]
        return torch.as_tensor(sum([self.orig_loss(g2t(groupedInputs, lbl), g2t(groupedTargets, lbl)) / len(g2t(groupedInputs, lbl)) if len(g2t(groupedInputs, lbl)) > 0 else 0 for lbl in labelTypes]))

# mean squared false error by Wang+16, IJCNN.
class MSFE_Loss(torch.nn.Module):
    def __init__(self, orig_loss):
        super(MSFE_Loss,self).__init__()

    def set_orig_loss(self, orig_loss):
        self.orig_loss = orig_loss

    def forward(self, x, y):
        #####
        assert False, 'MSFE_Loss is not defined yet.'
        total_loss = 0
        ######
        return total_loss

class cnn_lstm(nn.Module):

    def conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=np.int(np.floor((kernel_size-1)/2)), bias=False)

    def conv2d(self, in_channels, out_channels, kernel_size, stride):
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=np.int(np.floor((kernel_size-1)/2)), bias=False)

    def __init__(self, params, num_classes):
        super(cnn_lstm, self).__init__()
        self.params = params
        self.input_channelNum = 1
        self.stft_channelNum = 1
        self.rawDataDim = params.downsample_outputDim
        self.lstm_subseqLen = params.torch_lstm_length
        self.lstm_num_layers = params.torch_lstm_num_layers
        self.lstm_hidden_size = params.torch_lstm_hidden_size
        self.lstm_inputDim = params.torch_lstm_inputDim
        self.lstm_bidirectional = True if params.torch_lstm_bidirectional == 1 else False
        self.binNum4spectrum = round(params.wholeBand.getBandWidth() / params.binWidth4freqHisto)
        self.stft_time_bin_num = np.int(np.float(params.windowSizeInSec / params.stft_time_bin_in_seconds)) + 1   # plus one from edges

        if params.useSTFT:
            if params.useFreqHisto and params.useTime:
                # self.additionalFeatureDim = params.additionalFeatureDim
                # self.additionalFeatureDim = self.binNum4spectrum * self.stft_time_bin_num + 1
                self.additionalFeatureDim = params.outputDim_cnn_for_stft + 1
            elif params.useFreqHisto and not params.useTime:
                # self.additionalFeatureDim = self.binNum4spectrum * self.stft_time_bin_num
                self.additionalFeatureDim = params.outputDim_cnn_for_stft
            elif not params.useFreqHisto and params.useTime:
                self.additionalFeatureDim = 1
            else:
                self.additionalFeatureDim = 0
        else:
            if params.useFreqHisto and params.useTime:
                self.additionalFeatureDim = self.binNum4spectrum  + 1
            elif params.useFreqHisto and not params.useTime:
                self.additionalFeatureDim = self.binNum4spectrum
            elif not params.useFreqHisto and params.useTime:
                self.additionalFeatureDim = 1
            else:
                self.additionalFeatureDim = 0

        # if params.useRawData:
            # self.input_shape = (self.lstm_subseqLen, self.input_channelNum, self.rawDataDim + self.additionalFeatureDim)
        # else:
            # self.input_shape = (self.lstm_subseqLen, self.input_channelNum, self.additionalFeatureDim)
        self.filter_nums = params.torch_filter_nums
        self.kernel_sizes = params.torch_kernel_sizes
        self.strides = params.torch_strides
        self.filter_nums_for_stft = [8,8,8,8]
        self.kernel_sizes_for_stft = [3,3,3,3]
        self.strides_for_stft = [1,2,2,2]
        self.rawData_final_channel_num = self.filter_nums[-1]
        self.dropoutRate = params.dropoutRate
        self.skip_by = params.torch_skip_by
        self.layerNum = len(self.filter_nums)
        # self.stft_time_bin_in_seconds = params.stft_time_bin_in_seconds
        # self.avg_pool_size = params.torch_resnet_avg_pool_size
        # self.avg_pool = nn.AvgPool1d(self.avg_pool_size)
        # print('# in __init__, self.rawDataDim =', self.rawDataDim)
        # print('# in __init__, self.strides =', self.strides)

        # print('# in __init__, self.rawData_final_channel_num = ', self.rawData_final_channel_num, ', self.rawData_outputDim =', self.rawData_outputDim)
        # print('# in __init__, self.combined_size = ', self.combined_size)
        # self.batn_first = nn.BatchNorm1d(1)

        self.relu = nn.ReLU(inplace=True)
        in_channel_num = 1
        skip_in_channel_num = in_channel_num
        skip_inputDim = self.rawDataDim
        skip_in_blockID = 0
        batns = []
        convs = []
        drops = []
        skips = []
        for blockID, (out_channel_num, kernel_size, stride) in enumerate(zip(self.filter_nums, self.kernel_sizes, self.strides)):
            batns.append(nn.BatchNorm1d(in_channel_num))
            convs.append(self.conv(in_channel_num, out_channel_num, kernel_size, stride))
            drops.append(nn.Dropout(self.dropoutRate))
            in_channel_num = out_channel_num

            if blockID % self.skip_by == (self.skip_by - 1):
                # print('# blockID =', blockID)
                skip_out_layerID = blockID + 1
                skip_outputDim = reduce(lambda a, x: np.int(np.ceil((a - (x[0] - 1)) / x[1])), zip(self.kernel_sizes[skip_in_blockID:skip_out_layerID], self.strides[skip_in_blockID:skip_out_layerID]), skip_inputDim)
                skip_kernel_size = self.kernel_sizes[skip_in_blockID]
                skip_stride = reduce(lambda a, x: a * x, self.strides[skip_in_blockID:skip_out_layerID], 1)
                # print('# skip_inputDim =', skip_inputDim, ', skip_kernel_size =', skip_kernel_size)
                # print('# skip_outputDim =', skip_outputDim, ', skip_stride =', skip_stride)
                skip_out_channel_num = out_channel_num
                skips.append(self.conv(skip_in_channel_num, skip_out_channel_num, kernel_size=skip_kernel_size, stride=skip_stride))
                skip_in_channel_num = skip_out_channel_num
                skip_inputDim = skip_outputDim
                skip_in_blockID = skip_out_layerID
        self.batns = nn.ModuleList(batns)
        self.convs = nn.ModuleList(convs)
        self.drops = nn.ModuleList(drops)
        self.skips = nn.ModuleList(skips)

        #-----------
        # cnn for freq part
        if params.useSTFT:
            batns_for_stft = []
            convs_for_stft = []
            drops_for_stft = []
            in_channel_num = self.stft_channelNum
            for blockID, (out_channel_num, kernel_size, stride) in enumerate(zip(self.filter_nums_for_stft, self.kernel_sizes_for_stft, self.strides_for_stft)):
                batns_for_stft.append(nn.BatchNorm2d(in_channel_num))
                convs_for_stft.append(self.conv2d(in_channel_num, out_channel_num, kernel_size, stride))
                drops_for_stft.append(nn.Dropout(self.dropoutRate))
                in_channel_num = out_channel_num
            self.batns_for_stft = nn.ModuleList(batns_for_stft)
            self.convs_for_stft = nn.ModuleList(convs_for_stft)
            self.drops_for_stft = nn.ModuleList(drops_for_stft)

        # self.rawData_outputDim = reduce(lambda a, x: np.int(np.ceil((a - (x[0] - 1)) / x[1])), zip(self.kernel_sizes, self.strides), self.rawDataDim)
        self.rawData_outputDim = reduce(lambda a, x: np.int(np.ceil(a / x[1])), zip(self.kernel_sizes, self.strides), self.rawDataDim)
        # print('$$$ self.additionalFeatureDim =', self.additionalFeatureDim)
        if self.params.useRawData:
            self.combined_size = (self.rawData_final_channel_num * self.rawData_outputDim) + self.additionalFeatureDim
        else:
            self.combined_size = self.additionalFeatureDim
        # print('$$$ self.combined_size =', self.combined_size)

        # print('%%% in deepClassifier.py, combined_size =', self.combined_size)
        self.batn_combined = nn.BatchNorm1d(self.combined_size)
        self.dropout_combined = nn.Dropout(self.dropoutRate)
        self.final_fc_no_lstm = nn.Linear(self.combined_size, num_classes)
        if self.params.networkType == 'cnn_lstm':
            self.lstm = nn.LSTM(input_size=self.lstm_inputDim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional, dropout=self.dropoutRate, batch_first=True)
            ### self.lstm = nn.GRU  (input_size=self.lstm_inputDim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional, dropout=self.dropoutRate, batch_first=True)
            self.num_of_directions = 2 if self.lstm_bidirectional else 1
            self.fulc_combined_lstm = nn.Linear(self.combined_size, self.lstm_inputDim)
            self.final_fc_lstm = nn.Linear(self.lstm_hidden_size * self.num_of_directions, num_classes)

    def forward(self, x):
        # print('$%$%$%$%$% in forward, x.shape = (batchSize, seqLen, channelNum, featureNum) = ', x.shape)
        # normalized = self.batn_first(x)
        # connect two parts
        params = self.params
        if params.networkType == 'cnn_lstm':
            batchSize, subseqLen, channelNum, featureNum = x.shape
        else:
            batchSize, channelNum, featureNum = x.shape
            subseqLen = 1
        # print('in forward(), before reshape: x.shape =', x.shape)
        x = x.reshape(batchSize * subseqLen, channelNum, featureNum)
        # print('$%$%$%$%$% in forward, after reshape, x.shape = (new_batchSize, channelNum, featureNum) = ', x.shape)

        def cnn_on_stft(stft):
            # print('in cnn_on_stft(): stft.shape =', stft.shape)
            stft = stft.reshape(batchSize * subseqLen, self.stft_channelNum, self.binNum4spectrum, self.stft_time_bin_num)
            # print('%%% in forward(), after reshape: stft.shape =', stft.shape)
            for batn, conv, drop in zip(self.batns_for_stft, self.convs_for_stft, self.drops_for_stft):
                # print('  stft.shape =', stft.shape)
                stft = self.relu(conv(batn(stft)))
                if self.dropoutRate > 0:
                    stft = drop(stft)
            return stft

        if not params.useRawData:
            if params.useFreqHisto:
                if params.useSTFT:
                    # print('in forward() : x.shape =', x.shape)
                    freqFeature = cnn_on_stft(x[:,:,self.rawDataDim:-1])
                else:
                    freqFeature = x[:,:,self.rawDataDim:-1]

                if params.useTime:
                    combined = cat((freqFeature.view(freqFeature.size(0), -1), x[:,:,-1].view(x.size(0), -1)), dim=1)
                else:
                    combined = freqFeature.view(freqFeature.size(0), -1)
            elif params.useTime:
                combined = x[:,:,-1].view(x.size(0), -1)
            else:
                combined = None
        else:
            rawDataPart = x[:,:,:self.rawDataDim]
            # print('%%% rawDataPart.shape =', rawDataPart.shape)
            rawDataCopy = rawDataPart
            skipOriginLayer = 0
            skipID = 0
            for blockID in range(len(self.convs)):
                # print('  rawDataPart.shape =', rawDataPart.shape)
                rawDataPart = self.batns[blockID](rawDataPart)
                rawDataPart = self.convs[blockID](rawDataPart)
                rawDataPart = self.relu(rawDataPart)
                if self.dropoutRate > 0:
                    rawDataPart = self.drops[blockID](rawDataPart)

                if blockID % self.skip_by == (self.skip_by - 1):
                    rawDataCopy = self.skips[skipID](rawDataCopy)
                    rawDataPart = rawDataPart + rawDataCopy
                    rawDataCopy = rawDataPart
                    skipDestinLayer = blockID + 1
                    ### print('$%$%$%$%$%$%$%$ skip connection from layer ', skipOriginLayer, ' to layer ', skipDestinLayer, sep='')
                    skipOriginLayer = skipDestinLayer
                    skipID += 1
            if params.useFreqHisto:
                # print('$$$ in forward() : x.shape =', x.shape)
                # print('$$$ self.rawDataDim =', self.rawDataDim)
                if params.useSTFT:
                    freqFeature = cnn_on_stft(x[:,:,self.rawDataDim:-1])
                else:
                    freqFeature = x[:,:,self.rawDataDim:-1]

                # print('in forward(), before cat for combine, freqFeature.shape =', freqFeature.shape)
                if params.useTime:
                    combined = cat((rawDataPart.view(rawDataPart.size(0), -1), freqFeature.view(freqFeature.size(0), -1), x[:,:,-1].view(x.size(0), -1)), dim=1)
                else:
                    combined = cat((rawDataPart.view(rawDataPart.size(0), -1), freqFeature.view(freqFeature.size(0), -1)), dim=1)
            elif params.useTime:
                combined = cat((rawDataPart.view(rawDataPart.size(0), -1), x[:,:,-1].view(x.size(0), -1)), dim=1)
            else:
                combined = rawDataPart.view(rawDataPart.size(0), -1)

        # print('in forward(), combined.shape = ', combined.shape)
        combined = self.batn_combined(combined)
        if self.dropoutRate > 0:
            combined = self.dropout_combined(combined)
        # print('# before fulc, combined.shape =', combined.shape)
        # print('# after fulc, combined.shape =', combined.shape)
        # print('# in forward, sequences.shape =', sequences.shape)
        # print('# in forward, sequences =', sequences)
        if params.networkType == 'cnn_lstm':
            combined = self.fulc_combined_lstm(combined)
            sequences = combined.reshape(batchSize, subseqLen, -1)
            lstm_output, _ = self.lstm(sequences)
            # print('# in forward, lstm_output.shape = (batchSize, subseqLen, lstm_outputDim) ', lstm_output.shape)
            last_element_of_lstm_output = lstm_output[:,-1,:]
            # print('# in forward, last_element_of_lstm_output.shape = (batchSize, lstm_outputDim) ', last_element_of_lstm_output.shape)
            final_output = self.final_fc_lstm(last_element_of_lstm_output)
        else:
            final_output = self.final_fc_no_lstm(combined)
        # print('# final_output.shape = (batchSize, classNum) = ', final_output.shape)
        return final_output

'''
class simpleCNNWithFreqHistoAndTime(nn.Module):

    def conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=np.int(np.floor((kernel_size-1)/2)), bias=False)

    def __init__(self, params, num_classes):
        super(simpleCNNWithFreqHistoAndTime, self).__init__()
        self.params = params
        self.input_channelNum = 1
        self.rawDataDim = params.downsample_outputDim
        if params.useFreqHisto and params.useTime:
            self.additionalFeatureDim = params.additionalFeatureDim
        elif params.useFreqHisto and not params.useTime:
            self.additionalFeatureDim = params.additionalFeatureDim - 1
        elif not params.useFreqHisto and params.useTime:
            self.additionalFeatureDim = 1
        else:
            self.additionalFeatureDim = 0
        if params.useRawData:
            self.input_shape = (self.input_channelNum, self.rawDataDim + self.additionalFeatureDim)
        else:
            self.input_shape = (self.input_channelNum, self.additionalFeatureDim)
        self.filter_nums = params.torch_filter_nums
        self.kernel_sizes = params.torch_kernel_sizes
        self.strides = params.torch_strides
        self.rawData_final_channel_num = self.filter_nums[-1]
        self.dropoutRate = params.dropoutRate
        self.skip_by = params.torch_skip_by
        self.layerNum = len(self.filter_nums)
        # self.avg_pool_size = params.torch_resnet_avg_pool_size
        # self.avg_pool = nn.AvgPool1d(self.avg_pool_size)
        # print('# in __init__, self.rawDataDim =', self.rawDataDim)
        # print('# in __init__, self.strides =', self.strides)

        # print('# in __init__, self.rawData_final_channel_num = ', self.rawData_final_channel_num, ', self.rawData_outputDim =', self.rawData_outputDim)
        # print('# in __init__, self.combined_size = ', self.combined_size)
        # self.batn_first = nn.BatchNorm1d(1)

        self.relu = nn.ReLU(inplace=True)
        in_channel_num = 1
        skip_in_channel_num = in_channel_num
        skip_inputDim = self.rawDataDim
        skip_in_blockID = 0
        batns = []
        convs = []
        skips = []
        for blockID, (out_channel_num, kernel_size, stride) in enumerate(zip(self.filter_nums, self.kernel_sizes, self.strides)):
            batns.append(nn.BatchNorm1d(in_channel_num))
            convs.append(self.conv(in_channel_num, out_channel_num, kernel_size, stride))
            in_channel_num = out_channel_num

            if blockID % self.skip_by == (self.skip_by - 1):
                # print('# blockID =', blockID)
                skip_out_layerID = blockID + 1
                skip_outputDim = reduce(lambda a, x: np.int(np.ceil((a - (x[0] - 1)) / x[1])), zip(self.kernel_sizes[skip_in_blockID:skip_out_layerID], self.strides[skip_in_blockID:skip_out_layerID]), skip_inputDim)
                skip_kernel_size = self.kernel_sizes[skip_in_blockID]
                skip_stride = reduce(lambda a, x: a * x, self.strides[skip_in_blockID:skip_out_layerID], 1)
                # print('# skip_inputDim =', skip_inputDim, ', skip_kernel_size =', skip_kernel_size)
                # print('# skip_outputDim =', skip_outputDim, ', skip_stride =', skip_stride)
                skip_out_channel_num = out_channel_num
                skips.append(self.conv(skip_in_channel_num, skip_out_channel_num, kernel_size=skip_kernel_size, stride=skip_stride))
                skip_in_channel_num = skip_out_channel_num
                skip_inputDim = skip_outputDim
                skip_in_blockID = skip_out_layerID
        self.batns = nn.ModuleList(batns)
        self.convs = nn.ModuleList(convs)
        self.skips = nn.ModuleList(skips)

        # self.rawData_outputDim = reduce(lambda a, x: np.int(np.ceil((a - (x[0] - 1)) / x[1])), zip(self.kernel_sizes, self.strides), self.rawDataDim)
        self.rawData_outputDim = reduce(lambda a, x: np.int(np.ceil(a / x[1])), zip(self.kernel_sizes, self.strides), self.rawDataDim)
        if self.params.useRawData:
            self.combined_size = (self.rawData_final_channel_num * self.rawData_outputDim) + self.additionalFeatureDim
        else:
            self.combined_size = self.additionalFeatureDim

        # print('in deepClassifier.py, combined_size =', self.combined_size)
        self.batn_combined = nn.BatchNorm1d(self.combined_size)
        self.fulc_combined = nn.Linear(self.combined_size, num_classes)
        self.dropout_raw = nn.Dropout(self.dropoutRate)
        self.dropout_combined = nn.Dropout(self.dropoutRate)

    def forward(self, x):
        # print('$%$%$% in forward, x.shape =', x.shape)
        # normalized = self.batn_first(x)
        # connect two parts
        if not self.params.useRawData:
            additionalFeaturePart = x
            combined = additionalFeaturePart.view(additionalFeaturePart.size(0), -1)
        else:
            rawDataPart = x[:,:,:self.rawDataDim]
            rawDataCopy = rawDataPart
            skipID = 0
            for blockID in range(len(self.convs)):
                rawDataPart = self.batns[blockID](rawDataPart)
                rawDataPart = self.convs[blockID](rawDataPart)
                rawDataPart = self.relu(rawDataPart)

                if self.dropoutRate > 0:
                    rawDataPart = self.dropout_raw(rawDataPart)

                if blockID % self.skip_by == (self.skip_by - 1):
                    rawDataCopy = self.skips[skipID](rawDataCopy)
                    rawDataPart = rawDataPart + rawDataCopy
                    rawDataCopy = rawDataPart
                    skipID += 1
            # rawDataPart = self.avg_pool(rawDataPart)
            if self.additionalFeatureDim > 0:
                additionalFeaturePart = x[:,:,self.rawDataDim:]
                combined = cat((rawDataPart.view(rawDataPart.size(0), -1), additionalFeaturePart.view(additionalFeaturePart.size(0), -1)), dim=1)
            else:
                combined = rawDataPart.view(rawDataPart.size(0), -1)

        # print('%%% in forward(), combined.shape = ', combined.shape)
        combined = self.batn_combined(combined)
        if self.dropoutRate > 0:
            combined = self.dropout_combined(combined)
        final_output = self.fulc_combined(combined)
        # print('# final_output.shape = ', final_output.shape)
        return final_output

class ResidualBlock(nn.Module):

    def conv_by3(self, in_channels, out_channels, stride=1):
        return nn.Conv1d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=False)

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # print('# in_channels =', in_channels, ', out_channels =', out_channels)
        self.conv1 = self.conv_by3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv_by3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        # print('%%% x.shape =', x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet model
class ResNetWithFreqHistoAndTime(nn.Module):

    def conv_by3(self, in_channels, out_channels, stride=1):
        return nn.Conv1d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=False)

    def __init__(self, params, block, num_classes):
        super(ResNetWithFreqHistoAndTime, self).__init__()
        self.params = params
        self.input_channelNum = 1
        self.rawDataDim = params.downsample_outputDim
        self.additionalFeatureDim = params.additionalFeatureDim
        self.input_shape = (self.input_channelNum
        , self.rawDataDim + self.additionalFeatureDim)
        self.resnet_layer_nums = params.torch_resnet_layer_nums
        self.conv_channels = params.torch_resnet_conv_channels
        self.output_channels_coeffs = params.torch_resnet_output_channels_coeffs
        self.resblock_stride_nums = params.torch_resnet_resblock_stride_nums
        self.avg_pool_size = params.torch_resnet_avg_pool_size
        self.dropoutRate = params.dropoutRate
        # self.bn0 = nn.BatchNorm1d(1,eps=np.double(1e-05),momentum=np.double(0.1),track_running_stats=False)
        self.bn0 = nn.BatchNorm1d(1)
        self.first_conv = self.conv_by3(1, self.conv_channels)
        self.bn1 = nn.BatchNorm1d(self.conv_channels)
        self.relu = nn.ReLU(inplace=True)

        self.resblocks = []
        # print('% resblock = []')
        for coeff, res_layer, stride in zip(self.output_channels_coeffs, self.resnet_layer_nums, self.resblock_stride_nums):
            new_block, self.conv_channels = self.make_layer(block, self.conv_channels * coeff, res_layer, stride)
            # print('  appending to resblocks', new_block)
            self.resblocks.append(new_block)

        self.avg_pool = nn.AvgPool1d(self.avg_pool_size)

        self.rawData_outputDim = np.int(np.floor(self.rawDataDim / (reduce(mul, self.resblock_stride_nums) * self.avg_pool_size)))
        self.combined_size = (self.conv_channels * self.rawData_outputDim) + self.additionalFeatureDim
        self.bn_combined = nn.BatchNorm1d(self.combined_size)
        # print('combined_size =', self.combined_size)
        self.fc_combined = nn.Linear(self.combined_size, num_classes)
        # print('rawDataDim = ' + str(self.rawDataDim))
        # print('additionalFeatureDim = ' + str(self.additionalFeatureDim))

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.conv_channels != out_channels):
            downsample = nn.Sequential(
                self.conv_by3(self.conv_channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels))
        local_layers = []
        local_layers.append(block(self.conv_channels, out_channels, stride, downsample))
        # print('blocks: ', blocks)
        for i in range(1, blocks):
            local_layers.append(block(out_channels, out_channels))
        return nn.Sequential(*local_layers), out_channels

    def forward(self, x):
        normalized = self.bn0(x)
        rawDataPart = normalized[:,:,:self.rawDataDim]
        additionalFeaturePart = normalized[:,:,self.rawDataDim:]
        # print('# rawDataPart.shape =', rawDataPart.shape)

        # constructs a network for the additional feature part by FCN
        rawData_prior = rawDataPart
        rawDataPart = self.first_conv(rawDataPart)
        rawDataPart = self.bn1(rawDataPart)
        rawDataPart = self.relu(rawDataPart)
        for resblock in self.resblocks:
            # print('%%% adding', resblock)
            rawDataPart = resblock(rawDataPart)
        rawDataPart = self.avg_pool(rawDataPart)

        # connect two parts
        combined = cat((rawDataPart.view(rawDataPart.size(0), -1), additionalFeaturePart.view(additionalFeaturePart.size(0), -1)), dim=1)
        # print('# combined.shape = ', combined.shape)
        combined = self.bn_combined(combined)

        if self.dropoutRate > 0:
            combined = nn.Dropout(p=self.dropoutRate)(combined)

        # combined = rawDataPart
        # combined = combined.view(combined.size(0), -1)
        # print('# combined.shape　after view = ', combined.shape)
        final_output = self.fc_combined(combined)

        # print('# final_output.shape = ', final_output.shape)
        return final_output
'''

# Data loader
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, covariates, labels, transform=None):
        self.transform = transform
        self.covariates = covariates
        self.labels = labels
        # print('#$#$#$#$ in EEGDataset, covariates.shape =', covariates.shape)
        # self.labels = np.array([np.where(item == 1)[0][0] for item in labels])
        # print('# after where, self.labels.shape =', self.labels.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # print('#$#$#$#$# in EEGDataset.__getitem__, self.covariates[idx].shape =', self.covariates[idx].shape)
        # print('#$#$#$#$# in EEGDataset.__getitem__, self.labels[idx].shape =', self.labels[idx].shape)
        return (self.covariates[idx], self.labels[idx])

class DeepClassifier():

    #----------------------------
    # initialization
    def __init__(self, classLabels, classifierID, paramsForDirectorySetup, paramsForNetworkStructure):
        self.classLabels = classLabels
        self.paramsForDirectorySetup = paramsForDirectorySetup
        self.paramsForNetworkStructure = paramsForNetworkStructure
        self.weight_dir = paramsForDirectorySetup.pickledDir
        self.weight_path_best = self.weight_dir + '/weights.' + classifierID + '.pkl'
        self.rawDataDim = paramsForNetworkStructure.downsample_outputDim
        self.maximumStageNum = paramsForNetworkStructure.maximumStageNum
        self.stageLabel2stageID = paramsForNetworkStructure.stageLabel2stageID
        self.additionalFeatureDim = paramsForNetworkStructure.additionalFeatureDim
        # self.model_checkpoint_path = self.weight_dir + '/checkpoint.h5'
        self.optimizerType = paramsForNetworkStructure.optimizerType
        self.networkType = paramsForNetworkStructure.networkType
        self.adam_learningRate = paramsForNetworkStructure.adam_learningRate
        self.sgd_learningRate = paramsForNetworkStructure.sgd_learningRate
        self.deep_epochs = paramsForNetworkStructure.deep_epochs
        self.torch_patience = paramsForNetworkStructure.torch_patience
        self.batch_size = paramsForNetworkStructure.deep_batch_size
        self.dropoutRate = paramsForNetworkStructure.dropoutRate
        # self.dropoutRate = self.paramsForNetworkStructure.dropoutRate
        # self.deep_skipConnectionLayerNum = self.paramsForNetworkStructure.deep_skipConnectionLayerNum
        # self.stageLabels4evaluation = paramsForNetworkStructure.stageLabels4evaluation
        self.best_accuracy = 0
        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_str)
        if paramsForNetworkStructure.torch_loss_function == 'mfe':
            criterion = MFE_Loss()
            criterion.set_orig_loss(nn.CrossEntropyLoss())
            self.criterion = criterion
        elif paramsForNetworkStructure.torch_loss_function == 'msfe':
            criterion = MSFE_Loss()
            criterion.set_orig_loss(nn.CrossEntropyLoss())
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def generateModel(self):
        params = self.paramsForNetworkStructure
        print('in deepClassifier.generateModel, params.networkType =', params.networkType)
        # if params.extractorType == 'rawDataWithSTFTWithTime':
        if params.extractorType.startswith('rawData'):
            if params.networkType == '1dcnn':
                print('using 1DCNN for raw data')
                model = self.generate1DCNNForRawDataWithFCNForFreqHistoWithTime(params)
            elif params.networkType == 'simple_cnn':
                print('using simple CNN for raw data')
                model = self.generateCNNLSTM(params)
            elif params.networkType == 'cnn_lstm':
                print('using CNN-LSTM for raw data')
                model = self.generateCNNLSTM(params)
            else:
                print('using EmptyModel')
                model = self.generateEmptyModel(params)
        print("compiling the model")
        return model

    def generateCNNLSTM(self, params):
        model = cnn_lstm(params, len(self.classLabels))
        return model

    '''
    def generateSimpleCNNWithFCNForFreqHistoWithTime(self, params):
        model = simpleCNNWithFreqHistoAndTime(params, len(self.classLabels))
        return model

    def generate1DCNNForRawDataWithFCNForFreqHistoWithTime(self, params):
        model = ResNetWithFreqHistoAndTime(params, ResidualBlock, len(self.classLabels))
        # print('% model.__dict__ =', model.__dict__)
        return model
    '''

    # define fit(), since PyTorch model doesn't have one
    def fit(self, model, optimizer, train_loader, val_loader, epochs, batch_size):
        print('starting fit()')
        # self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            # print('using model.cuda()')
            torch.cuda.device(self.device)
            model.cuda()

        ### print('$%$%$%$ in fit(), calling summary() with model.input_shape =', model.input_shape)
        ### summary(model, model.input_shape, device=self.device_str)
        trainer = create_supervised_trainer(model, optimizer, self.criterion, device=self.device_str)
        metrics = {
            'accuracy':Accuracy(),
            'nll':Loss(self.criterion),
            'cm':ConfusionMatrix(num_classes=len(self.classLabels))
        }
        training_history = {'accuracy':[],'loss':[]}
        validation_history = {'accuracy':[],'loss':[]}
        # last_epoch = []
        evaluator = create_supervised_evaluator(model, metrics=metrics, device=self.device_str)

        def score_function(engine):
            val_loss = engine.state.metrics['nll']
            return - val_loss

        if self.torch_patience > 0:
            early_stopping = EarlyStopping(patience=self.torch_patience, score_function=score_function, trainer=trainer)
            evaluator.add_event_handler(Events.COMPLETED, early_stopping)

        # self.writer = SummaryWriter(log_dir=self.weight_dir)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            model.eval()
            # print('# running log_training_results(trainer):')
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            loss = metrics['nll']
            accuracy = metrics['accuracy']
            # self.writer.add_scalar('Loss/train', loss, self.n_iter)
            # self.writer.add_scalar('Accuracy/train', accuracy, self.n_iter)
            # last_epoch.append(0)
            training_history['accuracy'].append(accuracy)
            training_history['loss'].append(loss)
            print("Training - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
                  .format(trainer.state.epoch, accuracy, loss))
            model.train()

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            model.eval()
            # print('# running log_validation_results(trainer):')
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            loss = metrics['nll']
            accuracy = metrics['accuracy']
            # self.writer.add_scalar('Loss/test', loss, self.n_iter)
            # self.writer.add_scalar('Accuracy/test', accuracy, self.n_iter)
            validation_history['accuracy'].append(accuracy)
            validation_history['loss'].append(loss)
            print("Validation Results - Epoch: {}  Avg val accuracy: {:.4f} Avg val loss: {:.4f}"
                  .format(trainer.state.epoch, accuracy, loss))
            # save the model with the best accuracy
            if self.best_accuracy < accuracy:
                if not isdir(self.weight_dir):
                    mkdir(self.weight_dir)
                torch.save(model.state_dict(), self.weight_path_best)
                print('--> At Epoch: ', trainer.state.epoch, ', saved to ', self.weight_path_best, sep='')
                self.model = model
                self.best_accuracy = accuracy
            model.train()

        '''
        @trainer.on(Events.COMPLETED)
        def log_confusion_matrix(trainer):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            cm = metrics['cm']
            cm = cm.numpy()
            cm = cm.astype(int)
            classes = self.classLabels
            fig, ax = plt.subplots(figsize=(10,10))
            ax= plt.subplot()
            sns.heatmap(cm, annot=True, ax = ax,fmt="d")
            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(classes,rotation=90)
            ax.yaxis.set_ticklabels(classes,rotation=0)
        '''

        checkpointer = ModelCheckpoint(self.weight_dir, 'modelCheckpoint', save_interval=1, n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'epoch': model})

        # self.n_iter = 0
        print('before trainer.run()')
        trainer.run(train_loader, max_epochs=self.deep_epochs)
        # self.writer.close()

    def train(self, featuresBySamples, labels):
          print('len(featuresBySamples) =', len(featuresBySamples))
          # print('np.array(featuresBySamples).shape() =', np.array(featuresBySamples).shape())
          # print('np.array(labels).shape() =', np.array(labels).shape())
          validationRatio = 0.1

          #--------
          # converts labels to integers
          oneHots = np.array([stageLabel2oneHot(label, self.maximumStageNum, self.stageLabel2stageID) for label in labels])
          # print('oneHots.shape = ' + str(oneHots.shape))
          scalar_labels = np.array([np.where(item == 1)[0][0] for item in oneHots])
          # [print('oneHot = ' + str(oneHot)) for oneHot in oneHots]

          #---------
          # Train the Model.
          # print('starts training in fit():')
          model = self.generateModel()

          if self.optimizerType == 'Adam':
             optimizer = optim.Adam(model.parameters(), lr=self.adam_learningRate)
          elif self.optimizerType == 'SGD':
             optimizer = optim.SGD(model.parameters(), lr=self.sgd_learningRate)
          else:
             print('Error: Optimizer is not supported')

          # print('featuresBySamples.shape = ' + str(featuresBySamples.shape))
          featuresBySamples = featuresBySamples
          if len(featuresBySamples.shape) == 4:    # params.networkType == 'cnn_lstm'
             # featureTensor = featuresBySamples.transpose([0,2,1])
             featureTensor = featuresBySamples
          elif len(featuresBySamples.shape) == 3:    # params.networkType == 'simple_cnn'
             # featureTensor = featuresBySamples.transpose([0,2,1])
             featureTensor = featuresBySamples
          else:
             featureTensor = np.array([featuresBySamples]).transpose([1,0,2])

          featureTensor = featureTensor.astype(np.float)
          featureTensor = torch.autograd.Variable(torch.tensor(featureTensor, dtype=torch.float)).to(self.device)
          print('featureTensor.shape =', featureTensor.shape)

          if validationRatio > 0:
              sampleNum = len(labels)
              trainIDends = np.int(np.floor(sampleNum * (1-validationRatio)))
              trainIDs = range(0,trainIDends)
              valIDs = range(trainIDends+1,sampleNum)
              train_data = featureTensor[trainIDs]
              val_data = featureTensor[valIDs]
              # train_labels = oneHots[trainIDs]
              # val_labels = oneHots[valIDs]
              train_labels = scalar_labels[trainIDs]
              val_labels = scalar_labels[valIDs]

              print('for EEGDataset, train_data.shape =', train_data.shape)
              print('for EEGDataset, train_labels.shape =', train_labels.shape)
              train_data_with_labels = EEGDataset(train_data, train_labels)
              val_data_with_labels = EEGDataset(val_data, val_labels)
              print('self.batch_size =', self.batch_size)
              train_loader = torch.utils.data.DataLoader(dataset=train_data_with_labels, batch_size=self.batch_size, shuffle=True)
              val_loader = torch.utils.data.DataLoader(dataset=val_data_with_labels, batch_size=self.batch_size, shuffle=False)
          else:
              train_data = featureTensor
              # train_labels = oneHots
              train_labels = scalar_labels

              train_data_with_labels = EEGDataset(train_data, train_labels)
              train_loader = torch.utils.data.DataLoader(dataset=train_data_with_labels, batch_size=self.batch_size, shuffle=True)

          print("train_data.shape = " + str(train_data.shape) + ', train_labels.shape = ' + str(train_labels.shape))
          # print('type(train_data) =', type(train_data))
          # print('type(train_data[0,0,0]) =', type(train_data[0,0,0]))

          # convert data format if using a network with multiple input_shape
          # if self.paramsForNetworkStructure.extractorType == 'rawDataWithFreqHistoWithTime':
              # rawDataDim = self.paramsForNetworkStructure.downsample_outputDim
              # additionalFeatureDim = self.paramsForNetworkStructure.additionalFeatureDim
              # rawData = train_data[:,:rawDataDim,:]
              # additionalFeature = train_data[:,rawDataDim:,0]
              # train_data = [rawData, additionalFeature]
              # print('in deepClassifier.fit():')
              # print('rawData.shape = ' + str(rawData.shape))
              # print('additionalFeature.shape = ' + str(additionalFeature.shape))
              # val_rawData = val_data[:,:rawDataDim,:]
              # val_additionalFeature = val_data[:,rawDataDim:,0]
              # val_data = [val_rawData, val_additionalFeature]

          # print("transformed train_data.shape = " + str(train_data.shape) + ', train_labels.shape = ' + str(train_labels.shape))
          if validationRatio > 0:
              # print('self.paramsForNetworkStructure.deep_epochs = ', self.paramsForNetworkStructure.deep_epochs)
            self.fit(model, optimizer, train_loader, val_loader,
                epochs=self.paramsForNetworkStructure.deep_epochs,
                batch_size=self.batch_size)
          else:
            self.fit(model, optimizer, train_loader,
                epochs=self.paramsForNetworkStructure.deep_epochs,
                batch_size=self.batch_size)
            '''
              if self.paramsForNetworkStructure.deep_steps_per_epoch > 0:
                    self.fit(model, optimizer, train_loader, val_loader,
                        epochs=self.paramsForNetworkStructure.deep_epochs,
                        # steps_per_epoch=self.paramsForNetworkStructure.deep_steps_per_epoch,
                        batch_size=self.batch_size)
              else:
                    self.fit(model, optimizer, train_loader, val_loader,
                        epochs=self.paramsForNetworkStructure.deep_epochs,
                        # steps_per_epoch=self.paramsForNetworkStructure.deep_steps_per_epoch,
                        batch_size=self.batch_size)
          else:
              if self.paramsForNetworkStructure.deep_steps_per_epoch > 0:
                    self.fit(model, optimizer, train_loader,
                        epochs=self.paramsForNetworkStructure.deep_epochs,
                        # steps_per_epoch=self.paramsForNetworkStructure.deep_steps_per_epoch,
                        batch_size=self.batch_size)
              else:
                    self.fit(model, optimizer, train_loader,
                        epochs=self.paramsForNetworkStructure.deep_epochs,
                        batch_size=self.batch_size)
          '''

          print('The best model is saved at ', self.weight_path_best, ' with val_acc = ', self.best_accuracy, sep='')
          '''
          # save the model
          if not isdir(self.weight_dir):
              mkdir(self.weight_dir)
          print('writing to', self.model_path_final)
          # torch.save(model.state_dict(), self.model_path_final)
          torch.save(model, self.model_path_final)
          self.model = model
          '''

    def load_weights(self, weight_path):
        print('loading weights in deepClassifier.py from', weight_path)
        '''
        self.model = self.generateModel()
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.weight_path_best))
        else:
            self.model.load_state_dict(torch.load(self.weight_path_best, map_location='cpu'))
        '''
        self.model = self.generateModel()
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(weight_path))
        else:
            self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))

    #-----------------------
    def predict(self, featuresBySamples):
        # with self.graph.as_default():
        featuresBySamples = featuresBySamples.astype(np.double)
        ### featuresBySamples = torch.from_numpy(featuresBySamples).double()
        with torch.no_grad():

            featureTensor = featuresBySamples

            if self.networkType == 'cnn_lstm' and len(featureTensor.shape) == 3:
                featureTensor = np.array([featureTensor])
            if self.networkType != 'cnn_lstm' and len(featureTensor.shape) == 2:
                featureTensor = np.array([featureTensor]).transpose([1,0,2])
            # print('featureTensor.shape =', featureTensor.shape)
            self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(self.device_str)
            featureTensor = torch.tensor(featureTensor, dtype=torch.float).to(self.device)

            # print('# featureTensor.shape =', featureTensor.shape)
            self.model.eval()
            pred_loader = torch.utils.data.DataLoader(dataset=featureTensor, batch_size=1)
            # for feature in pred_loader:
                # print('feature.shape =', feature.shape)
                # pred_labels = self.model(feature.to(self.device))
            pred_labels = [self.model(feature).cpu().numpy() for feature in pred_loader][0]

        return pred_labels
