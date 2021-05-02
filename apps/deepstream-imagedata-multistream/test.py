#!/usr/bin/env python3
import pickle
import os
from os import walk
import cv2



def log_error(msg, _quit=True):
    print("-- PARAMETER ERROR --\n"*5)
    print(" %s \n" % msg)
    print("-- PARAMETER ERROR --\n"*5)
    if _quit:
        quit()
    else:
        return False


def file_exists(file_name):
    try:
        with open(file_name) as f:
            return file_name
    except OSError as e:
        return False


def file_exists_and_not_empty(file_name):
    if file_exists(file_name) and os.stat(file_name).st_size > 0:
        return True
    return False


def file_exists_and_empty(file_name):
    if file_exists(file_name) and os.stat(file_name).st_size == 0:
        return True
    return False


def read_images_in_dir(path_to_read):
    dir_name, subdir_name, file_names = next(walk(path_to_read))
    images = [item for item in file_names if '.jpeg' in item[-5:] or '.jpg' in item[-4:] or 'png' in item[-4:] ]


def delete_pickle(data_file):
    os.remove(data_file)
    if file_exists(data_file):
        raise Exception('unable to delete file: %s' % file_name)


def write_to_pickle(data, data_file):
    with open(data_file, mode='ab') as f:
        pickle.dump(data, f)
        print("saving into file...", data_file)


def read_pickle_2(pickle_file, exception=True):
    data = []
    try:
        with open(pickle_file, 'rb') as f:
            while True:
                try:
                    d = pickle.load(f)
                    data.append(d)
                except Exception as e:
                    break
            return len(data), data
    except OSError as e:
        if exception:
            com.log_error("Unable to open pickle_file: {}, original exception {}".format(pickle_file, str(e)))
        else:
            return 0, []


def read_pickle(pickle_file, exception=True):
    try:
        with open(pickle_file, 'rb') as f:
            known_face_encodings, known_face_metadata = pickle.load(f)
            return len(known_face_metadata), known_face_encodings, known_face_metadata
    except OSError as e:
        if exception:
            com.log_error("Unable to open pickle_file: {}, original exception {}".format(pickle_file, str(e)))
        else:
            return 0, [], []


pwd = os.getcwd()
metadata_file = 'data/video_encoded_faces/test_video_default_metadata.dat'
encodings_file = 'data/video_encoded_faces/test_video_default_encodings.dat'
test = pwd + '/data/video_encoded_faces/test_video_default.data'
test = '/tmp/read_from_directory/read_from_directory.dat'

t, datos, meta = read_pickle(test)
#datos = read_pickle(encodings_file)
numero = len(datos)
#print('numero:',numero,'\ndatos:\n', datos, 'meta', meta)
#print('numero:',numero,'\nmetadatos:\n', meta)
print('numero:',numero)
#print('meta 1:', meta[0]['face_image'])
#quit()
 
i = 0
for m in meta:
    img = m['face_image']
    cv2.imwrite("/tmp/stream_9/frame_" + str(i) + ".jpg", img)
    i += 1
quit()

for m in meta:
    i = 0 
    for img in m['face_image']:
        cv2.imwrite("/tmp/stream_99/frame_" + m['name'] + '_' + str(i) + ".jpg", img)
        print(m['name'],' confidence:', m['confidence'], ' difference ', m['difference'] )
        i += 1

