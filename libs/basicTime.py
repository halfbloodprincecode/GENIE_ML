import time
import datetime

def getTimeHR(timestamp=None, split: str =' ', dateFormat: str ='%Y-%m-%d', timeFormat: str ='%H:%M:%S', now: bool =False):
    if now:
        return datetime.datetime.now().strftime(f'{dateFormat}{split}{timeFormat}')
    else:    
        timestamp = timestamp if timestamp else time.time()
        return datetime.datetime.fromtimestamp(timestamp).strftime(f'{dateFormat}{split}{timeFormat}')