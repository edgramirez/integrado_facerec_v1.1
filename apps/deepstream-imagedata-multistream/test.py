import pickle
import os
from os import walk


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


def read_pickle(pickle_file, exception=True):
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

metadata_file = 'data/video_encoded_faces/test_video_default_metadata.dat'
encodings_file = 'data/video_encoded_faces/test_video_default_encodings.dat'
'''
images_file = '/tmp/images.data'

if file_exists(metadata_file):
    delete_pickle(metadata_file)
if file_exists(images_file):
    delete_pickle(images_file)
if file_exists(encodings_file):
    delete_pickle(encodings_file)

known_face_encodings = [1,1,1,1,1,1,1,1]
known_face_metadata = [1,1,1,1,1,1,1,1]
known_face_image = [1,1,1,1,1,1,1,1]
write_to_pickle(known_face_encodings, metadata_file)
write_to_pickle(known_face_metadata, images_file)
write_to_pickle(known_face_image, encodings_file)

known_face_encodings = [2,2,2,2,2,2,2,2]
known_face_metadata = [2,2,2,2,2,2,2,2]
known_face_image = [2,2,2,2,2,2,2,2]
write_to_pickle(known_face_encodings, metadata_file)
write_to_pickle(known_face_metadata, images_file)
write_to_pickle(known_face_image, encodings_file)

known_face_encodings = [3,3,3,3,3,3,3,3]
known_face_metadata = [3,3,3,3,3,3,3,3]
known_face_image = [3,3,3,3,3,3,3,3]
write_to_pickle(known_face_encodings, metadata_file)
write_to_pickle(known_face_metadata, images_file)
write_to_pickle(known_face_image, encodings_file)
'''

datos = read_pickle(metadata_file)
#datos = read_pickle(encodings_file)
numero = len(datos)
print('numero:',numero,'\ndatos:\n', datos)
