import numpy as np

headerLineNum = 22
finalEpoch = 2160
pred_fp = open('../data/0523/2020-05-23_11-29-28_pred.txt')
test_fp = open('../data/0523/0523natsu_TG.csv')

lines = list(test_fp)
# header_lines = lines[:headerLineNum]
# [print(line, end='') for line in header_lines]
test_lines = lines[headerLineNum+1:]
### test_lines = lines[headerLineNum+2:]
test_labels_orig = [line.split(',')[2] for line in test_lines][:finalEpoch]

pred_labels_orig = [line.rstrip() for line in pred_fp][:finalEpoch]
pred_labels = np.array(['S' if elem == '1' else elem for elem in pred_labels_orig])
test_labels = np.array(['S' if elem == 'NR' else elem for elem in test_labels_orig])

# print('len(pred_labels) =', len(pred_labels))
# print('len(test_labels) =', len(test_labels))
# print('pred_labels[:100] =', pred_labels[:100])
# print('test_labels[:100] =', test_labels[:100])
print('#(S,S) =', np.sum([p == 'S' and t == 'S' for p, t in zip(pred_labels, test_labels)]))
matching = [p == t for p, t in zip(pred_labels, test_labels)]
# matching = (pred_labels == test_labels)
# print('matching =', matching)
# print('correctNum =', correctNum)
correctNum = sum(matching)
print('precision =', correctNum / len(pred_labels))
