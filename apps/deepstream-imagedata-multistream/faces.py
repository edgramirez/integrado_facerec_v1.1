#!/usr/bin/python3
import sys
import lib.common as com

param_length = len(sys.argv)

msg = 'Usage: ' + sys.argv[0] + ' loadFaces | readVideo | readSilence | findImg | findVideo | compareData | appendTo'

if param_length < 2:
    com.log_error(msg)

if sys.argv[1] == 'loadFaces':
    if param_length == 2:
        known_faces = 'data/load'
        data_file = 'data/encoded_known_faces/knownFaces.dat'
    elif param_length == 4 and sys.argv[3] == 'output':
        known_faces = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import lib.biblioteca as biblio 
    biblio.encode_known_faces(known_faces, data_file)
elif sys.argv[1] == 'appendTo':
    if param_length == 2:
        known_faces = 'data/load'
        data_file = 'data/encoded_known_faces/knownFaces.dat'
    elif param_length == 4 and sys.argv[3] == 'output':
        known_faces = sys.argv[2]
        data_file = sys.argv[4]
    else:
        com.log_error(msg)

    import lib.biblioteca as biblio 
    biblio.encode_known_faces(known_faces, data_file, False)
elif sys.argv[1] == 'findImg':
    if param_length == 2:
        image_dir = 'data/find'
        data_file = 'data/encoded_known_faces/knownFaces.dat'
    elif param_length == 5 and sys.argv[3] == 'input':
        image_dir = sys.argv[2]
        data_file = sys.argv[4]
    else:
        com.log_error(msg)

    import lib.biblioteca as biblio 
    biblio.compare_pickle_against_unknown_images(data_file, image_dir)
elif sys.argv[1] == 'readVideo':
    if param_length == 2:
        video_input = 'data/video/test_video.mp4'
        data_file = 'data/video_encoded_faces/test_video_default.data'
    elif param_length == 5 and sys.argv[3] == 'input':
        image_dir = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import lib.biblioteca as biblio 
    kwargs = {}
    biblio.read_video(video_input, data_file, **kwargs)
elif sys.argv[1] == 'readSilence':
    if param_length == 2:
        video_input = 'data/video/test_video.mp4'
        data_file = 'data/video_encoded_faces/test_video_default.data'
    elif param_length == 5 and sys.argv[3] == 'input':
        image_dir = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import lib.biblioteca as biblio 
    kwargs = {'silence': True}
    biblio.read_video(video_input, data_file, **kwargs)
elif sys.argv[1] == 'findVideo':
    if param_length == 2:
        video_input = 'data/video/test_video.mp4'
        data_file = 'data/encoded_known_faces/knownFaces.dat'
    elif param_length == 5 and sys.argv[3] == 'known_data':
        image_dir = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import lib.biblioteca as biblio
    kwargs = {'find': True}
    biblio.read_video(video_input, data_file, **kwargs)
elif sys.argv[1] == 'compareData':
    if param_length == 2:
        video_data_file = 'data/video_encoded_faces/test_video_default.data'
        known_data_file = 'data/encoded_known_faces/knownFaces.dat'
    elif param_length == 5 and sys.argv[3] == 'known_data':
        image_dir = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import lib.biblioteca as biblio
    biblio.compare_data(video_data_file, known_data_file)
else:
    com.log_error(msg)
