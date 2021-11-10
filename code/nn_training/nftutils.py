import pickle

def read_file_in_list(path):
    with open(path) as (f):
        content = f.readlines()
    return content


def pickle_load(fn):
    file = open(fn, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def pickle_dump(data, fn):
    file = open(fn, 'wb')
    pickle.dump(data, file)
    file.close()


import time, os, stat

def file_age_in_seconds(pathname):
    return time.time() - os.stat(pathname)[stat.ST_MTIME]
