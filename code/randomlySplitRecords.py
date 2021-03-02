from __future__ import print_function
import sys
import numpy as np
import pickle
import random
import string
from parameterSetup import ParameterSetup
from fileManagement import getFileIDs

#---------------
# main
splitID = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
print('splitID =', splitID)

args = sys.argv
blockNum = int(args[1])

params = ParameterSetup()
# printMetadata(params)
prefix = 'eegAndStage'
fileIDs = getFileIDs(params.pickledDir, prefix)

recordNum = len(fileIDs)
print('recordNum =', recordNum)
# blockSize = np.int(np.floor(recordNum / blockNum))
blockSize = np.int(np.ceil(recordNum / blockNum))
print('blockSize =', blockSize)
print('fileIDs =', fileIDs[:5])
random.shuffle(fileIDs)
print('shuffled fileIDs =', fileIDs[:5])

blocks = []
for blockID in range(blockNum):
     blocks.append(fileIDs[(blockID * blockSize):((blockID + 1) * blockSize)])

with open(params.pickledDir + '/blocks_of_records.' + splitID + '.csv', 'w') as f:
# with open(params.pickledDir + '/blocks_of_records.csv', 'w') as f:
    for block in blocks:
        print('')
        print(block)
        f.write(','.join(block) + '\n')

print('splitID =', splitID)
