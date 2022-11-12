import os, glob
from libs.basicIO import ls

datadir = '/home/alihejrati/Desktop/untitled folder'

filelist = glob.glob(os.path.join(datadir, '**', '*.txt'))
filelist = [os.path.relpath(p, start=datadir) for p in filelist]
filelist = sorted(filelist)

L = sorted(ls(datadir, '**/*.txt', full_path=True))

print(filelist)
print(L)
print(filelist==L)