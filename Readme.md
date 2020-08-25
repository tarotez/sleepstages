## Required packages

In addition to installing Anaconda3, some Python packages should be installed. These can be installed using pip (Python package installer).

```
pip install pyserial torch pytorch-ignite torchsummary PyDAQmx
```

Version information:

```
Anaconda3 version 5.3.1 or higher
Python packages:
pyserial 3.4
torch 1.2.0
pytorch-ignite 0.3.0
torchsummary 1.5.1
PyDAQmx 1.4.2
```

To obtain signals from DAQ, NI-DAQmx should be downloaded from the National Instruments website and installed. This is not necessary when classifying signals offline.

http://www.ni.com/download/ni-daqmx-base-15.0/5648/en/

------------------------------------------------------------------------
## Using the program

Installation:

Source code and data files can be downloaded from GitHub at:

```
git clone git@github.com/tarotez/sleepstages
```

The "sleep stages" repository contains the source code and text files setting up default hyperparameters. The directory structure is as follows:

```
sleepstages/
  code/
  data/
    pickled
    waves
    WAVEDIR/
      Raw/
      Judge/
```

For training by end-users, "WAVEDIR" is a directory with an arbitrary name in which text files containing EEG raw data signals and ground truth stage labels are stored. These are not necessary if the user is using an already trained classifier (provided here) and not training a new classifier. For example, a user might want to train a new classifier to make the system predict sleep stages for epochs with different lengths.
The training program reads this directory to train classifiers. The user can place more than one WAVEDIR directory under the "data" directory. In other words, the user can prepare a directory for each dataset.

The code in the "code" directory uses the file named "path.json" to find the "data" directory.
The "data/pickled" directory should contain "params.json" for setting up parameters for feature extraction, training, and prediction.
Details of the parameters are indicated in the README.md file.

Running:
The GUI for online prediction can be started from a terminal by

```
python app.py
```

The predicted sleep stage can be sent to an Arduino-based external device connected by USB. The system can be run offline without acquiring signals from DAQ by

```
python app.py o
```

or

```
python app_offline.py
```

The latter is an alias to the former.

The program obtains waves from a pickled file in the "data/pickled" directory whose name starts with "EEGAndStage".

------------------------------------------------------------------------
## Training and testing a classifier

A new classifier can be trained and validated by the following procedure:

```
python readOfflineEEGandStageLabels2pickle.py WAVEDIR
python extractFeatures.py
python trainClassifier.py
python computeTrainingError.py
python computeTestError.py
```

readOfflineEEGandStageLabels2pickle.py reads text files containing EEG raw data signals and ground truth stage labels from the WAVEDIR directory. It writes files starting with "EEGAndStage" into the "data/pickled" directory. These files are in Python's pickle format to enable faster access.

extractFeatures.py reads "EEGAndStage" files and write files starting with "features". These files contain feature vectors used for training classifiers.

trainClassifier.py reads "features" and writes files starting with "weights", "params", and "files_used_for_training". These files contain randomly generated six-character IDs (i.e., classifier IDs) in their file names.

The "weights" file contains weight parameters obtained from training. The "params" file is a copy of "params.json" in "data/pickled" that is intended to save the parameters used for training the classifier. "files_used_for_training" indicates which recordings were used for training that classifier. These files are excluded when testing the classifier.

------------------------------------------------------------------------
## Readme file for raw data texts:

Each file contains raw data from one ~6-h recording. In a file, each row corresponds to a time point sampled at 128 Hz. The columns show time stamp, EEG amplitude, EMG amplitude, ground truth label, UTSN label, and UTSN-L label, in this order, separated by commas. As each epoch is 10 seconds long, ground truth labels and predicted sleep stages are repeated for 1,280 time points (i.e., rows).
