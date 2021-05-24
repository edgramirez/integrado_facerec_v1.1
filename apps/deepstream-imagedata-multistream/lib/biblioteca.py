import pickle
import os
import cv2
import time
import requests
import socket
import fcntl
import struct
import json
from os import walk
import face_recognition
import numpy as np
import lib.common as com
from datetime import datetime, timedelta

global header
global srv_url

font = cv2.FONT_HERSHEY_SIMPLEX
srv_url = 'https://mit.kairosconnect.app/'
header = None

##### GENERIC FUNCTIONS


def log_error(msg, quit_program = True):
    print("-- PARAMETER ERROR --\n"*5)
    print(" %s \n" % msg)
    print("-- PARAMETER ERROR --\n"*5)
    if quit_program:
        quit()
    else:
        return False


def get_supported_actions():
    return ('GET', 'POST', 'PUT', 'DELETE')


def get_timestamp():
    return int(time.time() * 1000)


def set_header(token_file = None):
    if token_file is None:
        token_file = '../.token'
    global header
    if header is None:
        if isinstance(token_file, str):
            token_handler = com.open_file(token_file, 'r+')
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


def update_known_faces_encodings(new_encoding):
    global known_face_encodings
    known_face_encodings.append(new_encoding)


def update_known_faces_metadata(new_metadata):
    global known_face_metadata
    known_face_metadata.append(new_metadata)


def update_known_face_information(new_encoding, new_metadata):
    update_known_faces_encodings(new_encoding)
    update_known_faces_metadata(new_metadata)


def write_to_pickle(known_face_encodings, known_face_metadata, data_file):
    with open(data_file, mode='wb') as f:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, f)


def cleanup_tracking_list(tracking_list, tracking_absence_dict, max_value):
    indexes_to_delete =  [ key for key, value in tracking_absence_dict.items() if value > max_value]
    for delete_item in indexes_to_delete:
        tracking_list.remove(delete_item)
        tracking_absence_dict.pop(delete_item)
    return tracking_list, tracking_absence_dict


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


def clasify_to_known_and_unknown(frame_image, face_locations, **kwargs):
    find = kwargs.get('find', False)
    silence = kwargs.get('silence', False)

    # Encode image of the face 
    face_encodings = face_recognition.face_encodings(frame_image, face_locations)
    face_labels = []

    total_visitors, known_face_metadata, known_face_encodings = get_known_faces_db()
    program_action = get_action()
    output_file = get_output_file()

    for face_location, face_encoding in zip(face_locations, face_encodings):
        # check if this face is in our list of known faces.
        metadata = biblio.lookup_known_face(face_encoding, known_face_encodings, known_face_metadata)

        face_label = None
        # If we found the face, label the face with some useful information.
        if metadata:
            print('uno ya visto')
            time_at_door = datetime.now() - metadata['first_seen_this_interaction']
            face_label = f"{metadata['name']} {int(time_at_door.total_seconds())}s"
        else:  # If this is a new face, add it to our list of known faces
            if program_action == actions['read']:
                print('reading ... nuevo')
                face_label = "New visitor" + str(total_visitors) + '!!'
                total_visitors += 1

                # Add the new face to our known faces metadata
                known_face_metadata = biblio.register_new_face_2(known_face_metadata, frame_image, face_location, 'visitor' + str(total_visitors))
                # Add the face encoding to the list of known faces
                known_face_encodings.append(face_encoding)

                if program_action == actions['read']:
                    cv2.imwrite("/tmp/stream_0/visitor_" + str(total_visitors)+".jpg", frame_image)
                    #biblio.write_to_pickle(known_face_encodings, known_face_metadata, output_file, False)

        if face_label is not None:
            face_labels.append(face_label)


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


#def new_face_metadata(face_image, source_info):
def new_face_metadata(face_image, name = None, camera_id = None, confidence = None, print_name = False):
    """
    Add a new person to our list of known faces
    """
    #camera_id = source_info.get('camera_id', None)
    #confidence = source_info.get('confidence', None)
    #source_type = source_info.get('source_type', None)

    if name is None:
        #source_info['name'] = source_info['camera_id'] + '_' + source_info['source_type'] + '_' + str(get_timestamp())
        name = camera_id + '_' + str(get_timestamp())
    else:
        if print_name:
            print('Saving face: {}'.format(name))

    today_now = datetime.now()

    return {
        'name': name,
        'face_id': 0,
        'camera_id': camera_id,
        'first_seen': today_now,
        'first_seen_this_interaction': today_now,
        'image': face_image,
        'confidence': confidence,
        'last_seen': today_now,
        'seen_count': 1,
        'seen_frames': 1
    }
    #    'source_type': source_type,


def delete_pickle(data_file):
    os.remove(data_file)
    if com.file_exists(data_file):
        raise Exception('unable to delete file: %s' % file_name)


def get_timestamp():
    return int(time.time() * 1000)


def lookup_known_face(face_encoding, known_face_encodings, known_face_metadata, tolerated_difference = 0.59):
    '''
    - See if this face was already stored in our list of faces
    - tolerated_difference: is the parameter that indicates how much 2 faces are similar, 0 is the best match and 1 means are completly different
    '''
    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if known_face_encodings:
        # Only check if there is a match
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if True in matches:
            # si hay un True en la lista entonces hay un match, get the indexes of these matches
            indexes = [ index for index, item in enumerate(matches) if item]

            # crear una lista dinamica con los indices que hicieron match
            only_true_known_face_encodings = [ known_face_encodings[ind] for ind in indexes ]

            # obtener la distancia de los elementos en la nueva lista contra el encoding de la nueva imagen
            face_distances = face_recognition.face_distance(only_true_known_face_encodings, face_encoding)

            # Get the match with the loggest distance to the image.
            best_match_index = np.argmin(face_distances)

            # La distancia de este elemento con la menor distancia tiene que ser menor a nuestro parametro de aceptacion
            if face_distances[best_match_index] < tolerated_difference:
                # Values returned:  
                #        meta que hace match,                     el indice real de la lista global, distancia a la imagen analizada
                return   known_face_metadata[indexes[best_match_index]], indexes[best_match_index], face_distances[best_match_index]

            return None, None, face_distances[best_match_index]
    return None, None, None


def encode_known_faces_from_images_in_dir(image_path, output_file):
    '''
    Esta funccion codifica los rostros encotrados en las imagenes presentes en el diretorio especificado
    '''
    files, root = com.read_images_in_dir(image_path)
    known_face_metadata = []
    known_face_encodings = []
    #source_type = com.source_type['IMAGE']

    write_to_file = False
    for file_name in files:
        # load the image into face_recognition library
        source_info = {}
        face_obj = face_recognition.load_image_file(root + '/' + file_name)
        name = os.path.splitext(file_name)[0]
        known_face_encodings, known_face_metadata = encode_and_update_face_image(face_obj, name, known_face_encodings, known_face_metadata)
        if known_face_encodings:
            write_to_file = True
    if write_to_file:
        write_to_pickle(known_face_encodings, known_face_metadata, output_file)
    else:
        print('Ningun archivo de imagen contine rostros: {}'.format(image_path))


def encode_and_update_face_image(face_obj, name, face_encodings, face_metadata):
    new_encoding, new_metadata = encode_face_image(face_obj, name, None, None, True)
    if new_encoding is not None:
        face_encodings.append(new_encoding)
        face_metadata.append(new_metadata)
    return face_encodings, face_metadata


def encode_face_image(face_obj, face_name, camera_id, confidence, print_name):
    # covert the array into cv2 default color format
    # THIS ALREADY DONE IN CROP
    #rgb_frame = cv2.cvtColor(face_obj, cv2.COLOR_RGB2BGR)

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
            face_metadata_dict = new_face_metadata(face_obj, face_name, camera_id, confidence, print_name)
            return encoding[0], face_metadata_dict

    #print('Ningun rostro detectado en: . {}'.format(name))
    return None, None


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


def compare_data(data_file, known_faces_data, tolerated_difference_list):
    # load data from binary db of all faces from video
    total_visitors, video_face_encodings, video_faces_metadata = read_pickle(data_file)

    # load data from binary db of known faces 
    total_known_faces, known_face_encodings, known_face_metadata = read_pickle(known_faces_data)

    if total_known_faces == 0 or total_visitors == 0:
        com.log_error("One of the db does not contain information {}")

    if not isinstance(tolerated_difference_list, list) and len(tolerated_difference_list) > 0:
        com.log_error("Paramter range_list must be 'list'. Current type {}".format(type(range_list)))

    for tolerated_difference in tolerated_difference_list:
        if tolerated_difference < 1 and tolerated_difference > 0:
            print('\n---- Using tolerated difference: {} ----'.format(tolerated_difference))
            #compare_data(face_encoding, known_face_encodings, known_face_metadata, tolerated_difference)

            #for known_face_encoding, known_metadata in zip(known_face_encodings, known_face_metadata):
            for video_face_encoding, video_metadata in zip(video_face_encodings, video_faces_metadata):
                # check one by one all the images in the video against the known faces
                #metadata, best_index, lowest_distances = lookup_known_face(known_face_encoding, video_face_encodings, video_faces_metadata, tolerated_difference)
                metadata, best_index, lowest_distances = lookup_known_face(video_face_encoding, known_face_encodings, known_face_metadata, tolerated_difference)
                if best_index:
                    #print(metadata)
                    print('-'*8)
                    print('Subject {} found'.format(metadata['name']))
                    #print('camera_id {}'.format(video_faces_metadata[best_index]['camera_id']))
                    print('initial {}'.format(video_faces_metadata[best_index]['first_seen']))
                    print('last {}'.format(video_faces_metadata[best_index]['last_seen']))
                    print('distance: {}'.format(lowest_distances))
    #quit()

    '''
        if best_index:
            #print('Face {} detected at {} {} {} {} {}'.format(metadata['name'], video_metadata['first_seen'], video_metadata['first_seen_this_interaction'], video_metadata['last_seen'], video_metadata['seen_count'], video_metadata['seen_frames']))
            print('Face {} '.format(metadata['name']))
    '''

def read_video(video_input, data_file, **kwargs):
    video_capture = cv2.VideoCapture(video_input)
    find = kwargs.get('find', False)
    silence = kwargs.get('silence', False)

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
                        known_face_metadata = edit_meta_face(known_face_metadata, face_image, 'visitor' + str(total_visitors))

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

