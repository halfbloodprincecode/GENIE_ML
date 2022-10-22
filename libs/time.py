import time
import datetime

def getTimeHR(timestamp=None, split=' ', dateFormat='%Y-%m-%d', timeFormat='%H:%M:%S'):
    timestamp = timestamp if timestamp else time.time()
    return datetime.datetime.fromtimestamp(timestamp).strftime(f'{dateFormat}{split}{timeFormat}')