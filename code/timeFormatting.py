import datetime
import re

def presentTimeEscaped():
    presentTime = str(datetime.datetime.now())
    presentTime = presentTime.replace(' ', '_')
    # presentTime = presentTime.replace('/', '-')
    # presentTime = '-'.join(presentTime.split('\/'))
    presentTime = re.sub(r'\:', '-', presentTime)
    presentTime = re.sub(r'\.[0-9]*$', '', presentTime)
    return presentTime
