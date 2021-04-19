import re
import os
import glob
import json
import time
import requests
import threading

import fcntl
import socket
import struct

from math import sqrt
from random import seed, randint
from datetime import datetime, timedelta


### There could be a if here to include this only if these elements are required
import pickle
import cv2
from os import walk
import face_recognition
import numpy as np



global first_time_set
global last_time_set
global header
global server_url
global sd_keys
global nfps
global people_counting_enabled
global aforo_enabled
global social_distance_enabled
global aforo_url
global social_distance_url
global people_counting_url
global plate_detection_url
global srv_url

srv_url = 'https://mit.kairosconnect.app/'
header = None

first_time_set = set()
last_time_set = set()


##### GENERIC FUNCTIONS


def log_error(msg, quit_program = True):
    print("-- PARAMETER ERROR --\n"*5)
    print(" %s \n" % msg)
    print("-- PARAMETER ERROR --\n"*5)
    if quit_program:
        quit()
    else:
        return False


def api_get_number_of_frames_per_second():
    '''
    TODO: function not yet defined
    '''
    return None


def file_exists(file_name):
    try:
        with open(file_name) as f:
            return file_name
    except OSError as e:
        return False


def open_file(file_name, option='a+'):
    if file_exists(file_name):
        return open(file_name, option)
    return False


def create_file(file_name, content = None):

    if file_exists(file_name):
        os.remove(file_name)
        if file_exists(file_name):
            raise Exception('unable to delete file: %s' % file_name)

    if content:
        with open(file_name, 'w+') as f:
            f.write(content)
    else:
        with open(file_name, 'w+') as f:
            f.close()

    return True


def get_number_of_frames_per_second():
    global nfps

    nfps = api_get_number_of_frames_per_second()

    if nfps is None:
        return 16

    return nfps


def get_supported_actions():
    return ('GET', 'POST', 'PUT', 'DELETE')


def get_timestamp():
    return int(time.time() * 1000)


def set_header(token_file = None):
    if token_file is None:
        token_file = '.token'

    global header

    if header is None:
        if isinstance(token_file, str):
            token_handler = open_file(token_file, 'r+')
            if token_handler:
                header = {'Content-type': 'application/json', 'X-KAIROS-TOKEN': token_handler.read().split('\n')[0]}
                print('Header correctly set')
                return True

    return False

def getHwAddr(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8927,  struct.pack('256s', bytes(ifname, 'utf-8')[:15]))
    return ':'.join('%02x' % b for b in info[18:24])


def get_ip_address(ifname):
    return [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]


def get_machine_macaddresses():
    list_of_interfaces = [item for item in os.listdir('/sys/class/net/') if item != 'lo']
    macaddress_list = []

    for iface_name in list_of_interfaces:
        ip = get_ip_address(iface_name)
        if ip:
            macaddress_list.append(getHwAddr(iface_name))
            return macaddress_list


def get_server_info(abort_if_exception = True, quit_program = True):
    global srv_url

    url = srv_url + 'tx/device.getConfigByProcessDevice'

    for machine_id in get_machine_macaddresses():
        machine_id = '00:04:4b:eb:f6:dd'  # HARDCODED MACHINE ID
        data = {"id": machine_id}
        
        if abort_if_exception:
            response = send_json(data, 'POST', url)
        else:
            options = {'abort_if_exception': False}
            response = send_json(data, 'POST', url, **options)

    if response:
        return json.loads(response.text)
    else:
        return log_error("Unable to retrieve the device configuration from the server. Server response".format(response), quit_program = quit_program)


def send_json(payload, action, url = None, **options):
    set_header()
    global header

    if action not in get_supported_actions() or url is None:
        raise Exception('Requested action: ({}) not supported. valid options are:'.format(action, get_supported_actions()))

    retries = options.get('retries', 2)
    sleep_time = options.get('sleep_time', 1)
    expected_response = options.get('expected_response', 200)
    abort_if_exception = options.get('abort_if_exception', True)

    data = json.dumps(payload)

    # emilio comenta esto para insertar en MongoDB
    # return True

    for retry in range(retries):
        try:
            if action == 'GET':
                r = requests.get(url, data=data, headers=header)
            elif action == 'POST':
                r = requests.post(url, data=data, headers=header)
            elif action == 'PUT':
                r = requests.put(url, data=data, headers=header)
            else:
                r = requests.delete(url, data=data, headers=header)
            return r
        except requests.exceptions.ConnectionError as e:
            time.sleep(sleep_time)
            if retry == retries - 1 and abort_if_exception:
                raise Exception("Unable to Connect to the server after {} retries\n. Original exception: {}".format(retry, str(e)))
        except requests.exceptions.HTTPError as e:
            time.sleep(sleep_time)
            if retry == retries - 1 and abort_if_exception:
                raise Exception("Invalid HTTP response in {} retries\n. Original exception: {}".format(retry, str(e)))
        except requests.exceptions.Timeout as e:
            time.sleep(sleep_time)
            if retry == retries - 1 and abort_if_exception:
                raise Exception("Timeout reach in {} retries\n. Original exception: {}".format(retry, str(e)))
        except requests.exceptions.TooManyRedirects as e:
            time.sleep(sleep_time)
            if retry == retries - 1 and abort_if_exception:
                raise Exception("Too many redirection in {} retries\n. Original exception: {}".format(retry, str(e)))


def check_if_object_is_in_area2(object_coordinates, reference_line, m, b):
    '''
    # returns True if object is in Area2
    # returns False if object is in Area1
    '''
    if m is None:
        if object_coordinates[0] > reference_line[0][0]:
            return True
        return False
    elif m == 0:
        if object_coordinates[1] > reference_line[0][1]:
            return True
        return False
    else:
        y_overtheline = (m * object_coordinates[0]) + b

        if object_coordinates[1] > y_overtheline:
            return True
        else:
            return False


def is_point_insde_polygon(x, y, polygon_length, polygon):

    p1x,p1y = polygon[0]
    for i in range(polygon_length+1):
        p2x,p2y = polygon[i % polygon_length]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        # returns True if x,y are inside
                        return True
        p1x,p1y = p2x,p2y

    # returns False if x,y are not inside
    return False


##### PEOPLE COUNTING

def set_service_people_counting_url():
    global people_counting_url, srv_url
    people_counting_url = srv_url + 'SERVICE_NOT_DEFINED_'


def people_counting(camera_id, total_objects):
    '''
    Sending only the total of detected objects
    '''
    global people_counting_url
    
    date = get_timestamp()
    alert_id = str(date) + '_' + str(camera_id) + '_' + str(date)
    data = {
            'id': alert_id,
            'camera-id': camera_id,
            '#total_updated_at': date,
            'object_id': total_objects,
            }
    #print('People_counting first time..POST', data, people_counting_url)
    #x = threading.Thread(target=send_json, args=(data, 'POST', srv_url))
    #x.start()


##### AFORO

def set_aforo_url():
    global aforo_url, srv_url
    aforo_url = srv_url + 'tx/video-people.endpoint'


def aforo(box, object_id, ids, camera_id, initial, last, entradas, salidas, outside_area=None, reference_line=None, m=None, b=None, rectangle=None):
    '''
    A1 is the closest to the origin (0,0) and A2 is the area after the reference line
    A1 is by default the outside
    A2 is by default the inside
    This can be changed by modifying the configuration variable "outside_area" to 2 (by default 1)
    x = box[0]
    y = box[1]

    initial -  must be a dictionary, and will be used to store the first position (area 1 or area2) of a given ID
    last -     must be a dictionary, and will be used to store the last position (area 1 or area2) of a given ID
    '''

    if rectangle:
        # si el punto esta afuera del area de interes no evaluamos
        if box[0] < rectangle[0] or box[0] > rectangle[4] or box[1] > rectangle[5] and box[1] < rectangle[1]:
            if reference_line:
                return entradas, salidas
            else:
                outside_area = 1
                area = 1

    if reference_line:
        if check_if_object_is_in_area2(box, reference_line, m, b):
            area = 2
        else:
            area = 1
    else:
        outside_area = 1
        area = 2

    if outside_area == 1:
        direction_1_to_2 = 1
        direction_2_to_1 = 0
    else:
        direction_1_to_2 = 0
        direction_2_to_1 = 1

    if object_id not in initial:
        initial.update({object_id: area})
        if object_id not in last:
            return entradas, salidas
    else:
        last.update({object_id: area})

    # De igual forma si los elementos continen las misma areas en el estado 
    # actual que en el previo, entonces no tiene caso evaluar mas
    if initial[object_id] == last[object_id]:
        return entradas, salidas

    for item in last.keys():
        if initial[item] == 1 and last[item] == 2:
            time_in_epoc = get_timestamp()
            data_id = str(time_in_epoc) + '_' + str(object_id)
            data = {
                    'id': data_id,
                    'direction': direction_1_to_2,
                    'camera-id': camera_id,
                    '#date-start': time_in_epoc,
                    '#date-end': time_in_epoc,
                }
            initial.update({item: 2})

            print('Sending Json of camera_id: ', camera_id, 'ID: ',item, 'Sal:0,Ent:1 = ', direction_1_to_2, "tiempo =",time_in_epoc)
            x = threading.Thread(target=send_json, args=(data, 'PUT', aforo_url,))
            x.start()

            if direction_1_to_2 == 1:
                entradas += 1
            else:
                salidas += 1

        elif initial[item] == 2 and last[item] == 1:
            time_in_epoc = get_timestamp()
            data_id = str(time_in_epoc) + '_' + str(object_id)
            data = {
                    'id': data_id,
                    'direction': direction_2_to_1,
                    'camera-id': camera_id,
                    '#date-start': time_in_epoc,
                    '#date-end': time_in_epoc,
                }
            initial.update({item: 1})

            print('Sending Json of camera_id: ', camera_id, 'ID: ',item, 'Sal:0,Ent:1 = ', direction_2_to_1, "tiempo =",time_in_epoc)
            x = threading.Thread(target=send_json, args=(data, 'PUT', aforo_url,))
            x.start()

            if direction_2_to_1 == 1:
                entradas += 1
            else:
                salidas += 1

    return entradas, salidas


##### SOCIAL DISTANCE

def set_social_distance_url():
    global social_distance_url, srv_url
    social_distance_url = srv_url + 'tx/video-socialDistancing.endpoint'


def social_distance2(camera_id, ids_and_boxes, tolerated_distance, persistence_time, max_side_plus_side, detected_ids):
    '''
    social distance is perform in pairs of not repeated pairs
    Being (A, B, C, D, E, F) the set of detected objects

    The possible permutation are:

       AB AC AD AE AF
          BC BD BE BF
             CD CE CF
                DE DF
                   Ef

    We are going to start compararing the first element (index=0 or i=0)
    '''
    # TODO: diccionario puede crecer mucho depurarlo comparando los elementos que dejen de existir o no sean detectados despues de 5seg')

    # sorting elements to always have the same evaluation order 
    ids = [ item for item in ids_and_boxes.keys() ]
    ids.sort()
    # creating the list 
    i = 1
    for pivot in ids[:-1]:
        for inner in ids[i:]:
            if pivot not in detected_ids:
                Ax = ids_and_boxes[pivot][0]
                x = ids_and_boxes[inner][0]
    
                if Ax > x:
                    dx = Ax -x
                else:
                    dx = x - Ax
    
                if dx < tolerated_distance:
                    Ay = ids_and_boxes[pivot][1]
                    y = ids_and_boxes[inner][1]

                    if Ay > y:
                        dy = Ay - y
                    else:
                        dy = y - Ay

                    if (dx + dy) < max_side_plus_side and sqrt((dx*dx) + (dy*dy)) < tolerated_distance:
                        # first time detection for pivot A and associated B
                        pivot_time = get_timestamp()
                        detected_ids.update({
                            pivot: {
                                inner:{
                                    '#detected_at': pivot_time,
                                    '#reported_at': None,
                                    'reported': False,
                                    }
                                }
                            })
            else:
                if inner not in detected_ids[pivot]:
                    Ax = ids_and_boxes[pivot][0]
                    x = ids_and_boxes[inner][0]
        
                    if Ax > x:
                        dx = Ax -x
                    else:
                        dx = x - Ax
        
                    if dx < tolerated_distance:
                        Ay = ids_and_boxes[pivot][1]
                        y = ids_and_boxes[inner][1]

                        if Ay > y:
                            dy = Ay - y
                        else:
                            dy = y - Ay

                        if (dx + dy) < max_side_plus_side and sqrt((dx*dx) + (dy*dy)) < tolerated_distance:
                            # firt time detection for associated C is registered
                            detected_at_inner = get_timestamp()
                            detected_ids[pivot].update({
                                inner:{
                                    '#detected_at': detected_at_inner,
                                    '#reported_at': None,
                                    'reported': False,
                                    }
                                })
                else:
                    Ax = ids_and_boxes[pivot][0]
                    x = ids_and_boxes[inner][0]
        
                    if Ax > x:
                        dx = Ax -x
                    else:
                        dx = x - Ax

                    if dx > tolerated_distance:
                        if not detected_ids[pivot][inner]['reported']:
                            del detected_ids[pivot][inner]
                    else:
                        Ay = ids_and_boxes[pivot][1]
                        y = ids_and_boxes[inner][1]

                        if Ay > y:
                            dy = Ay - y
                        else:
                            dy = y - Ay

                        if (dx + dy) >= max_side_plus_side or sqrt((dx*dx) + (dy*dy)) >= tolerated_distance:
                            del detected_ids[pivot][inner]
                        else:
                            current_time = get_timestamp()
                            initial_time = detected_ids[pivot][inner]['#detected_at']
                            if not detected_ids[pivot][inner]['reported'] and (current_time - initial_time) >= persistence_time:
                                detected_ids[pivot][inner].update({'#reported_at': current_time})
                                detected_ids[pivot][inner].update({'reported': True})
                                alert_id = str(current_time) + '_' +  str(pivot) + '_and_'+ str(inner)
                                data = {
                                    'id': alert_id,
                                    'camera-id': camera_id,
                                    '#date': current_time,
                                    }
                                print('Social distance', data, social_distance_url, 'PUT', 'distance=', sqrt((dx*dx) + (dy*dy)), 'tolerada:', tolerated_distance)
                                x = threading.Thread(target=send_json, args=(data, 'PUT', social_distance_url,))
                                x.start()
            i += 1


#### MASK DETECTION

def set_mask_detection_url():
    global mask_detection_url, srv_url
    mask_detection_url = srv_url + 'tx/video-maskDetection.endpoint'


def mask_detection(mask_id, no_mask_ids, camera_id, reported_class = 0):
    time_in_epoc = get_timestamp()
    data_id = str(time_in_epoc) + '_' + str(mask_id)
    data = {
        'id': data_id,
        'mask': reported_class,
        'camera-id': camera_id,
        '#date-start': time_in_epoc,
        '#date-end': time_in_epoc
        }

    print('Mask detection', data, mask_detection_url, 'PUT')
    x = threading.Thread(target=send_json, args=(data, 'PUT', mask_detection_url,))
    x.start()


#### PLATE DETECTION

def set_plate_detection_url():
    global plate_detection_url, srv_url
    plate_detection_url = srv_url + 'TO_BE_SETUP______tx/video-plateDetection.endpoint'


#### FACE DETECTION

font = cv2.FONT_HERSHEY_SIMPLEX


def read_pickle(pickle_file, exception=True):
    try:
        with open(pickle_file, 'rb') as f:
            known_face_encodings, known_face_metadata = pickle.load(f)
            return len(known_face_metadata), known_face_encodings, known_face_metadata
    except OSError as e:
        if exception:
            log_error("Unable to open pickle_file: {}, original exception {}".format(pickle_file, str(e)))
        else:
            return 0, [], []


def draw_box_around_face(face_locations, face_labels, image):
    # Draw a box around each face and label each face
    for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)


def display_recent_visitors_face(known_face_metadata, frame):
    number_of_recent_visitors = 0
    for metadata in known_face_metadata:
        # If we have seen this person in the last minute, draw their image
        if datetime.now() - metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 1:
            # Draw the known face image
            x_position = number_of_recent_visitors * 150
            frame[30:180, x_position:x_position + 150] = metadata["face_image"]
            number_of_recent_visitors += 1

            # Label the image with how many times they have visited
            visits = metadata['seen_count']
            visit_label = f"{visits} visits"
            if visits == 1:
                visit_label = "First visit"
            cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)


def register_new_face(known_face_metadata, face_image, name):
    """
    Add a new person to our list of known faces
    """
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
    today_now = datetime.now()
    known_face_metadata.append({
        "first_seen": today_now,
        "first_seen_this_interaction": today_now,
        "last_seen": today_now,
        "seen_count": 1,
        "seen_frames": 1,
        "name": name,
        "face_image": face_image,
    })

    return known_face_metadata


def write_to_pickle(known_face_encodings, known_face_metadata, data_file, new_file = True):
    if new_file and file_exists(data_file):
        os.remove(data_file)
        if file_exists(data_file):
            raise Exception('unable to delete file: %s' % file_name)

        with open(data_file,'wb') as f:
            face_data = [known_face_encodings, known_face_metadata]
            pickle.dump(face_data, f)
            print("Known faces saved...")
    else:
        with open(data_file,'ab') as f:
            face_data = [known_face_encodings, known_face_metadata]
            pickle.dump(face_data, f)


def lookup_known_face(face_encoding, known_face_encodings, known_face_metadata, tolerance = 0.62):
    """
    See if this is a face we already have in our face list
    """
    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0 or float(tolerance) > 1 or float(tolerance) < 0:
        return None

    # Only check if there is a match
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    if True in matches:
        # If there is a match, then get the best distances only on the index with "True" to ignore the process on those that are False
        indexes = [ index for index, item in enumerate(matches) if item]
        only_true_known_face_encodings = [ known_face_encodings[ind] for ind in indexes ]

        face_distances = face_recognition.face_distance(only_true_known_face_encodings, face_encoding)
        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < tolerance:
            # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
            metadata = [ known_face_metadata[ind] for ind in indexes ]
            metadata = metadata[best_match_index]

            # Update the metadata for the face so we can keep track of how recently we have seen this face.
            metadata["last_seen"] = datetime.now()
            metadata["seen_frames"] += 1

            if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=5):
                metadata["first_seen_this_interaction"] = datetime.now()
                metadata["seen_count"] += 1

            return metadata

    return None


def encode_known_faces(known_faces_path, output_file, new_file = True):
    files, root = com.read_images_in_dir(known_faces_path)

    names = []
    known_face_encodings = []
    known_face_metadata = []

    for file_name in files:
        # load the image into face_recognition library
        face_obj = face_recognition.load_image_file(root + '/' + file_name)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = face_obj[:, :, ::-1]

        # try to get the location of the face if there is one
        face_location = face_recognition.face_locations(rgb_small_frame)

        # if got a face, loads the image, else ignores it
        if face_location:
            name = os.path.splitext(file_name)[0]
            names.append(name)

            # Grab the image of the face from the current frame of video
            top, right, bottom, left = face_location[0]
            face_image = rgb_small_frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (150, 150))

            encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(encoding)

            new_known_face_metadata = register_new_face(known_face_metadata, face_image, name)
    if names:
        print(names)
        write_to_pickle(known_face_encodings, new_known_face_metadata, output_file, new_file)
    else:
        print('Ningun archivo de imagen contine rostros')


def compare_pickle_against_unknown_images(pickle_file, image_dir):
    total_known_faces, known_face_encodings, known_face_metadata = read_pickle(pickle_file)

    files, root = com.read_images_in_dir(image_dir)
    for file_name in files:
        file_path = os.path.join(root, file_name)
        test_image = face_recognition.load_image_file(file_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

        # try to get the location of the face if there is one
        face_locations = face_recognition.face_locations(test_image)

        # if got a face, loads the image, else ignores it
        if face_locations:
            encoding_of_faces = face_recognition.face_encodings(test_image, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, encoding_of_faces):
                face_title = 'desconocido'
                metadata = lookup_known_face(face_encoding, known_face_encodings, known_face_metadata)
                if metadata:
                    face_title = metadata['name']

                cv2.rectangle(test_image, (left, top),(right, bottom),(0, 0, 255), 2)
                cv2.putText(test_image, face_title, (left, top-6), font, .75, (180, 51, 225), 2)

            cv2.imshow('Imagen', test_image)
            cv2.moveWindow('Imagen', 0 ,0)

            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
        else:
            print("Image to search does not contains faces")
            print(file_path)


def compare_data(data_file, known_faces_data):
    # load data from binary db of all faces from video
    total_visitors, video_face_encodings, video_faces_metadata = read_pickle(data_file)
    # load data from binary db of known faces 
    total_known_faces, known_face_encodings, known_face_metadata = read_pickle(known_faces_data)

    for video_face_encoding, video_metadata in zip(video_face_encodings, video_faces_metadata):
        # check one by one all the images in the video against the known faces
        metadata = lookup_known_face(video_face_encoding, known_face_encodings, known_face_metadata)

        if metadata:
            print('Face {} detected at {}'.format(
                metadata['name'],
                video_metadata['first_seen'],
                video_metadata['first_seen_this_interaction'],
                video_metadata['last_seen'],
                video_metadata['seen_count'],
                video_metadata['seen_frames']
                ))


def read_video(video_input, data_file, **kwargs):
    video_capture = cv2.VideoCapture(video_input)
    find =  kwargs.get('find', False)
    silence =  kwargs.get('silence', False)

    # Track how long since we last saved a copy of our known faces to disk as a backup.
    number_of_faces_since_save = 0

    # load data from binary db
    total_visitors, known_face_encodings, known_face_metadata = read_pickle(data_file, False)

    frame_counter = 0
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if not ret:
            break

        frame_counter += 1

        # Process image every other frame to speed up
        if frame_counter % 3 == 0:
            continue

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                # See if this face is in our list of known faces.
                metadata = lookup_known_face(face_encoding, known_face_encodings, known_face_metadata)

                face_label = None
                # If we found the face, label the face with some useful information.
                if metadata:
                    time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                    face_label = f"{metadata['name']} {int(time_at_door.total_seconds())}s"
                # If this is a brand new face, add it to our list of known faces
                else:
                    if not find:
                        face_label = "New visitor" + str(total_visitors) + '!!'
                        total_visitors += 1

                        # Grab the image of the the face from the current frame of video
                        top, right, bottom, left = face_location
                        face_image = small_frame[top:bottom, left:right]
                        face_image = cv2.resize(face_image, (150, 150))

                        # Add the new face to our known faces metadata
                        known_face_metadata = register_new_face(known_face_metadata, face_image, 'visitor' + str(total_visitors))

                        # Add the face encoding to the list of known faces
                        known_face_encodings.append(face_encoding)
                
                if face_label is not None:
                    face_labels.append(face_label)

            # Draw a box around each face and label each face
            if face_label is not None:
                draw_box_around_face(face_locations, face_labels, frame)

            # Display recent visitor images
            display_recent_visitors_face(known_face_metadata, frame)

        # Display the final frame of video with boxes drawn around each detected fames
        if not silence:
            cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if not silence and cv2.waitKey(1) & 0xFF == ord('q'):
            if not find:
                write_to_pickle(known_face_encodings, known_face_metadata, data_file)
            break

        # We need to save our known faces back to disk every so often in case something crashes.
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            if not find:
                write_to_pickle(known_face_encodings, known_face_metadata, data_file)
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


