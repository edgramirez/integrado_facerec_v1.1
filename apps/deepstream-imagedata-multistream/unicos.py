#!/usr/bin/env python3
import pickle
import os
from os import walk
import cv2
import face_recognition
from datetime import datetime, timedelta
import numpy as np


global total_visitors, known_face_encodings, known_face_metadata
known_face_encodings = []
known_face_metadata = []


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

def set_known_faces_db(total, encodings, metadata):
    global total_visitors, known_face_encodings, known_face_metadata
    total_visitors = total
    known_face_encodings = encodings
    known_face_metadata = metadata

def get_known_faces_db():
    global total_visitors, known_face_metadata, known_face_encodings
    return total_visitors, known_face_metadata, known_face_encodings

def encode_face_image(face_obj, name):
    # covert the array into cv2 default color format
    rgb_frame = cv2.cvtColor(face_obj, cv2.COLOR_RGB2BGR)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = face_obj[:, :, ::-1]

    # try to get the location of the face if there is one
    face_location = face_recognition.face_locations(rgb_small_frame)

    # if got a face, loads the image, else ignores it
    if face_location:
        # Grab the image of the face from the current frame of video
        top, right, bottom, left = face_location[0]
        face_image = rgb_small_frame[top:bottom, left:right]
        face_image = cv2.resize(face_image, (150, 150))
        encoding = face_recognition.face_encodings(face_image)

        # if encoding empty we assume the image was already treated 
        if len(encoding) == 0:
            encoding = face_recognition.face_encodings(rgb_small_frame)

        if encoding:
            face_metadata_dir = new_face_metadata(rgb_frame, name)
            #print('<---------return values\n',encoding[0], face_metadata_dir,'\n--------->\n')
            return encoding[0], face_metadata_dir

    print('Ningun rostro detectado en: . {}'.format(name))
    return None, None

def new_face_metadata(face_image, name):
    """
    Add a new person to our list of known faces
    """
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
    today_now = datetime.now()
    return {
        'name': name,
        'face_id': 0,
        'first_seen': today_now,
        'first_seen_this_interaction': today_now,
        'face_image': face_image,
        'confidence': 0,
        'last_seen': today_now,
        'seen_count': 1,
        'seen_frames': 1
    }

def lookup_known_face(face_encoding, known_face_encodings, known_face_metadata, difference = 0.43):
    """
    - See if this is a face we already have in our face list
    - Tolerance is the parameter that indicates how much 2 faces are similar, 0 is the best match and 1 means this 2 faces are completly different
    """
    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    #print('aqui')
    if known_face_encodings:
        print('prospecto nuevo')
        # Only check if there is a match
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if True in matches:
            print('match pero falta evaluar distacia')
            # If there is a match, then get the best distances only on the index with "True" to ignore the process on those that are False
            indexes = [ index for index, item in enumerate(matches) if item]

            only_true_known_face_encodings = [ known_face_encodings[ind] for ind in indexes ]
            face_distances = face_recognition.face_distance(only_true_known_face_encodings, face_encoding)

            # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
            best_match_index = np.argmin(face_distances)
            best_match_encoding = only_true_known_face_encodings[best_match_index]

            # si la distancia en muy chica es la misma persona
            if face_distances[best_match_index] < difference:
                print('coincidencia')
                only_true_known_face_metadata = [ known_face_metadata[ind] for ind in indexes ]
                return only_true_known_face_metadata[best_match_index], indexes[best_match_index], face_distances[best_match_index]
            return None, None, face_distances[best_match_index]

    return None, None, None

def update_known_faces_encodings(new_encoding):
    global known_face_encodings
    known_face_encodings.append(new_encoding)


def update_known_faces_metadata(new_metadata):
    global known_face_metadata
    known_face_metadata.append(new_metadata)


def update_known_face_information(new_encoding, new_metadata):
    update_known_faces_encodings(new_encoding)
    update_known_faces_metadata(new_metadata)

pwd = os.getcwd()
test = '/tmp/read_from_directory/read_from_directory.dat'

t, datos, meta = read_pickle(test)

set_known_faces_db(0, [], [])
i = 0
for m in meta:
    total_visitors, known_face_metadata, known_face_encodings = get_known_faces_db()
    print('ANALIZANDO')
    img = m['face_image']
    img_encoding, img_metadata = encode_face_image(img, m['name'])
    #print('-------------')
    #print(img_encoding, img_metadata)
    #print('-------------')
    try:
        if img_encoding is not None:
            metadata, best_index, difference = lookup_known_face(img_encoding, known_face_encodings, known_face_metadata)
            # si es None entonces es un elementos nuevo - Agregarloa known_face_encodings, known_face_metadata
            if metadata is None:
                print('NUEVO', difference)
                update_known_face_information(img_encoding, img_metadata)
                cv2.imwrite('/tmp/v2/leidos_' + str(i) + ".jpg", img)
            else:
                print('SEGUNDA VISTA', best_index, difference)
    except Exception as e:
        print('edgar', str(e))
        quit()
    #cv2.imwrite("/tmp/stream_9/frame_" + str(i) + ".jpg", img)
    i += 1
quit()

for m in meta:
    i = 0 
    for img in m['face_image']:
        cv2.imwrite("/tmp/stream_99/frame_" + m['name'] + '_' + str(i) + ".jpg", img)
        print(m['name'],' confidence:', m['confidence'], ' difference ', m['difference'] )
        i += 1

