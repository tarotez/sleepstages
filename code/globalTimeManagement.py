from __future__ import print_function

def getTimeDiffInSeconds(time1, time2):
    # print('time1 = ' + str(time1))
    # print('time2 = ' + str(time2))
    e1 = time1.split(':')
    e2 = time2.split(':')
    s1 = (((float(e1[0]) * 60) + float(e1[1])) * 60) + float(e1[2])
    s2 = (((float(e2[0]) * 60) + float(e2[1])) * 60) + float(e2[2])
    # print('timeInSec = ' + str(s1))
    return s2 - s1
