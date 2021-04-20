import pickle
import os
import cv2
from os import walk
import face_recognition
import numpy as np
import lib.common as com
from datetime import datetime, timedelta


font = cv2.FONT_HERSHEY_SIMPLEX


def write_to_pickle(known_face_encodings, known_face_metadata, data_file):
    with open(data_file, mode='wb') as f:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, f)
        print("Known faces saved.................................................... OK")


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
        "confidence": 0.0,
        "face_image": face_image,
    })

    return known_face_metadata


def delete_pickle(data_file):
    os.remove(data_file)
    if com.file_exists(data_file):
        raise Exception('unable to delete file: %s' % file_name)


def lookup_known_face(face_encoding, known_face_encodings, known_face_metadata, tolerance = 0.62):
    """
    See if this is a face we already have in our face list
    """
    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0 or float(tolerance) > 1 or float(tolerance) < 0:
        return None

    # Only check if there is a match
    try:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    except Exception as e:
        print(str(e))
        print(type(known_face_encodings),'\n', type(face_encoding),'\n')
        print(known_face_encodings, '\n\n', face_encoding)
        #help(face_recognition.compare_faces)
        quit()

    if True in matches:
        # If there is a match, then get the best distances only on the index with "True" to ignore the process on those that are False
        indexes = [ index for index, item in enumerate(matches) if item]
        print('indexes:', indexes)
        only_true_known_face_encodings = [ known_face_encodings[ind] for ind in indexes ]

        face_distances = face_recognition.face_distance(only_true_known_face_encodings, face_encoding)
        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < tolerance:
            # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
            #try:
            #    print('aqui...', known_face_metadata)
            #    metadata = [ known_face_metadata[ind] for ind in indexes ]
            #except Exception as e:
            #    print(str(e))
            #    quit()
            #metadata = [ known_face_metadata[ind] for ind in indexes ]
            #metadata = known_face_metadata[best_match_index]
            #print(metadata)
            #quit()

            # Update the metadata for the face so we can keep track of how recently we have seen this face.
            #metadata["last_seen"] = datetime.now()
            #metadata["seen_frames"] += 1

            #if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=5):
            #    metadata["first_seen_this_interaction"] = datetime.now()
            #    metadata["seen_count"] += 1

            #return metadata
            print('best_match_index: ', best_match_index)
            print(known_face_metadata[best_match_index])

            return known_face_metadata[best_match_index], best_match_index

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
        #write_to_pickle(known_face_encodings, new_known_face_metadata, output_file, new_file)
        write_to_pickle(known_face_encodings, new_known_face_metadata, output_file)
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

