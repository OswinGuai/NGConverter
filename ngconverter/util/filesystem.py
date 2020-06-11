
import os
import shutil

def remakedirs(directory, mode=int('0777', 8)):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    oldmask = os.umask(000)
    os.makedirs(directory, mode)
    os.umask(oldmask)
