#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import numpy as np
import pyds
import cv2
import os
import os.path
from os import path
import lib.biblioteca as biblio
import lib.common as com
import face_recognition
from datetime import datetime, timedelta


fps_streams={}
frame_count={}
saved_count={}
global PGIE_CLASS_ID_FACE
PGIE_CLASS_ID_FACE=0
global PGIE_CLASS_ID_MAKE
PGIE_CLASS_ID_MAKE=2

MAX_DISPLAY_LEN=64
PGIE_CLASS_ID_FACE = 0
PGIE_CLASS_ID_PLATE = 1
PGIE_CLASS_ID_MAKE = 2
PGIE_CLASS_ID_MODEL = 3
#MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_WIDTH=1280
#MUXER_OUTPUT_HEIGHT=1080
MUXER_OUTPUT_HEIGHT=720
MUXER_BATCH_TIMEOUT_USEC=4000000
#TILED_OUTPUT_WIDTH=1920
TILED_OUTPUT_WIDTH=1280
#TILED_OUTPUT_HEIGHT=1080
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"
pgie_classes_str= ["face", "Placa", "Marca","Modelo"]

CURRENT_DIR = os.getcwd()


global total_visitors
global known_face_encodings
global known_face_metadata
global action
global action_types
global known_faces_indexes
global not_applicable_id
global video_initial_time
global fake_frame_number
global found_faces
global tracking_absence_dict
global output_file

known_faces_indexes = []
not_applicable_id = []
known_face_metadata = []
known_face_encodings = []
found_faces = []
tracking_absence_dict = {}
output_file = {}
action_types = {'read': 1, 'find': 2, 'compare': 3}
action = {}
fake_frame_number = 0


### setters ###

def set_action(camera_id, value):
    global action_types

    if value in action_types:
        global action
        return action.update({camera_id: action_types[value]})

    com.log_error('Unable to set up value:{}, must be one of this: {}'.format(value, action_types.keys()))


def set_known_faces_db_name(camera_id, value):
    global input_file
    input_file = value


def set_output_db_name(camera_id, value):
    global output_file

    output_file.update({camera_id: value})


def set_known_faces_db(camera_id, total, encodings, metadata):
    global total_visitors, known_face_encodings, known_face_metadata

    total_visitors = total
    known_face_encodings = encodings
    known_face_metadata = metadata


def set_metadata(camera_id, metadata):
    global known_face_metadata

    known_face_metadata = metadata


def set_encoding(camera_id, encodings):
    global known_face_metadata
    known_face_encodings = encodings


def set_tracking_absence_dict(camera_id, dictionary):
    global tracking_absence_dict
    tracking_absence_dict = dictionary


def set_known_faces_indexes(camera_id, new_list = None):
    global known_faces_indexes
    if new_list:
        known_faces_indexes = new_list
    else:
        known_faces_indexes = []


def add_faces_encodings(face_encoding):
    global known_face_encodings
    known_face_encodings.append(face_encoding)


### getters ###
def get_action(camera_id):
    global action

    if camera_id in action:
        return action[camera_id]

    com.log_error('No value found for camera_id: {}'.format(camera_id))


def get_output_db_name(camera_id):
    global output_file

    if camera_id in output_file:
        return output_file[camera_id]

    com.log_error('No value found for camera_id: {}'.format(camera_id))


def get_known_faces_db(camera_id):
    global total_visitors, known_face_metadata, known_face_encodings
    return total_visitors, known_face_metadata, known_face_encodings


def get_known_faces_indexes(camera_id):
    global known_faces_indexes
    return known_faces_indexes


def get_not_applicable_id(camera_id):
    global not_applicable_id
    return not_applicable_id


def get_tracking_absence_dict(camera_id):
    global tracking_absence_dict
    return tracking_absence_dict


def get_found_faces(camera_id):
    global found_faces
    return found_faces


def get_camera_id(camera_id):
    return 'FA:KE:MA:C:AD:DR:ES:S9'


def get_delta(camera_id):
    return 3


def get_similarity(camera_id):
    return 0.7


def save_found_faces(camera_id, metadata_of_found_faces):
    global found_faces
    found_faces = metadata_of_found_faces


def crop_and_get_faces_locations(n_frame, obj_meta, confidence):
    # convert python array into numy array format.
    frame_image = np.array(n_frame, copy=True, order='C')

    # covert the array into cv2 default color format
    rgb_frame = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)

    # draw rectangle and crop the face
    crop_image = draw_bounding_boxes(rgb_frame, obj_meta, confidence)

    return crop_image


def add_new_face_metadata(face_image, name, confidence, difference, face_id):
    """
    Add a new person to our list of known faces
    """
    global known_face_metadata
    today_now = datetime.now()
    known_face_metadata.append({
        'name': name,
        'face_id': [face_id],
        'first_seen': today_now,
        'first_seen_this_interaction': today_now,
        'image': face_image,
        'confidence': [confidence],
        'difference': [difference],
        'last_seen': today_now,
        'seen_count': 1,
        'seen_frames': 1
    })
    return known_face_metadata


def register_new_face_3(camera_id, face_encoding, image, name, confidence, difference, face_id):
    # Add the new face metadata to our known faces metadata
    add_new_face_metadata(image, name, confidence, difference, face_id)
    # Add the face encoding to the list of known faces encodings
    add_faces_encodings(camera_id, face_encoding)
    # Add new element to the list - this list maps and mirrows the face_ids for the meta
    #add_new_known_faces_indexes(face_id)
    update_known_faces_indexes(camera_id, face_id)


def update_not_applicable_id(camera_id, new_value, best_index = None):
    global not_applicable_id
    if best_index is not None:
        not_applicable_id[best_index] = new_value
    else:
        # check value was not previously registered in list
        if new_value not in not_applicable_id:
            not_applicable_id.append(new_value)


def update_known_faces_indexes(camera_id, new_value, best_index = None):
    global known_faces_indexes
    if best_index is not None:
        known_faces_indexes[best_index] = new_value
    else:
        # check value was not previously registered in list
        if new_value not in known_faces_indexes:
            known_faces_indexes.append(new_value)


def classify_to_known_and_unknown(camera_id, image, obj_id, name, program_action, confidence, frame_number, delta, default_similarity, known_faces_indexes, known_face_metadata, known_face_encodings):
    # Using face_detection library, try to encode the image
    update = False
    best_index = None

    if program_action == action_types['read']:
        difference = None
        if obj_id in known_faces_indexes:
            best_index = known_faces_indexes.index(obj_id)
            if confidence - known_face_metadata[best_index]['confidence'][-1] > 0.0038:
                img_encoding, img_metadata = biblio.encode_face_image(image, name, camera_id, confidence, False)

                if img_encoding is None:
                    return False

                known_face_encodings[best_index] = img_encoding
                known_face_metadata[best_index]['image'] = image
                known_face_metadata[best_index]['confidence'].append(confidence)
                set_metadata(camera_id, known_face_metadata)
                set_encoding(camera_id, known_face_encodings)
                #cv2.imwrite("/tmp/cambio_id/nuevo_ind" + str(best_index) + ".jpg", image)
                #cv2.imwrite("/tmp/cambio_id/original_ind" + str(best_index) + ".jpg", known_face_metadata[best_index]['image'])
            update = True
        else:
            img_encoding, img_metadata = biblio.encode_face_image(image, name, camera_id, confidence, False)

            if img_encoding is None:
                return False

            metadata, best_index, difference = biblio.lookup_known_face(img_encoding, known_face_encodings, known_face_metadata)

            if best_index is not None:
                update = True
                if confidence > metadata['confidence'][-1]:
                    known_face_encodings[best_index] = img_encoding
                    known_face_metadata[best_index]['image'] = image
                    known_face_metadata[best_index]['confidence'].append(confidence)
                    set_metadata(camera_id, known_face_metadata)
                    set_encoding(camera_id, known_face_encodings)
                    #cv2.imwrite("/tmp/cambio_id/nuevo_" + str(best_index) + ".jpg", image)
                    #cv2.imwrite("/tmp/cambio_id/original_" + str(best_index) + ".jpg", metadata['image'])
            # TODO hay que reducir la lista cada minuto por que los ids que ya pasaron y que no aparecen ya no van a aparecer - mismo clenaup

        if update:
            today_now = datetime.now()
            if today_now - known_face_metadata[best_index]['last_seen'] > timedelta(seconds=delta) and known_face_metadata[best_index]['seen_frames'] > 1:
                known_face_metadata[best_index]['last_seen'] = today_now
                known_face_metadata[best_index]['seen_count'] += 1
                known_face_metadata[best_index]['seen_frames'] += 1
                set_metadata(camera_id, known_face_metadata)

                return True
            else:
                return False
        else:  
            img_encoding, img_metadata = biblio.encode_face_image(image, name, camera_id, confidence, False)

            if img_encoding is None:
                return False

            face_label = 'visitor_' + str(len(known_face_metadata))
            register_new_face_3(camera_id, img_encoding, image, face_label, confidence, None, obj_id)
            return True

            # TODO: remove this file writting cause is only for debug purposes
            #cv2.imwrite(folder_name + "/stream_" + str(frame_meta.pad_index) + "/frame_" + str(total_visitors) + ".jpg", image)
            #cv2.imwrite("/tmp/stream_0/frame_" + str(face_label) + ".jpg", image)

    else: # ---- FIND FACES ----
        not_applicable_id = get_not_applicable_id(camera_id)

        if obj_id not in known_faces_indexes:
            print('id: ', obj_id, known_faces_indexes, not_applicable_id)
            if obj_id in not_applicable_id: 
                return False

            img_encoding, img_metadata = biblio.encode_face_image(image, name, camera_id, confidence, False)

            if img_encoding is None:
                return  False

            metadata, best_index, difference = biblio.lookup_known_face(img_encoding, known_face_encodings, known_face_metadata)

            # hay coincidencia alguno de los rostros buscados
            if best_index is None:
                #update_not_applicable_id(camera_id, obj_id)
                return False

            # verificar si ya se encuentra detectado bajo otro id y entonces solo actualiza
            # obtine la estructura y datos actuales de los rostros encontrados 
            found_faces = get_found_faces(camera_id)
            today_now = datetime.now()
            i = 0
            for found_face in found_faces:
                if found_face['name'] == metadata['name']:
                    if today_now - found_faces[i]['last_seen'][-1] > timedelta(seconds=delta):
                        found_faces[i]['seen_count'] += 1

                    found_faces[i]['face_id'].append(obj_id)
                    found_faces[i]['seen_frames'].append(frame_number)
                    found_faces[i]['last_seen'].append(today_now)

                    if confidence - found_face['confidence'][-1] > 0.0038 and found_faces[i]['difference'][-1] > difference: 
                        print('Sujeto1 {}, encontrado en frame {} con id: {} ,confidence: n{}/o{}, distance: n{}/o{}'.format(metadata['name'], frame_number, obj_id, confidence, found_face['confidence'][-1], difference, found_faces[i]['difference'][-1]))
                        found_faces[i]['confidence'].append(confidence)
                        found_faces[i]['difference'].append(difference)
                        found_faces[i]['image'].append(image)

                    # actualiza las variables globales
                    print('Sujeto1.1 {}, encontrado en frame {} con id: {} ,confidence: n{}/o{}, distance: n{}/o{}'.format(metadata['name'], frame_number, obj_id, confidence, found_face['confidence'][-1], difference, found_faces[i]['difference'][-1]))
                    save_found_faces(camera_id, found_faces)
                    update_known_faces_indexes(camera_id, obj_id)
                    return True
                i += 1

            print('Sujeto1.2 {}, encontrado en frame {} con id: {} ,confidence: {}, distance: {}'.format(metadata['name'], frame_number, obj_id, confidence, difference))
            found_faces.append({
                'name': metadata['name'],
                'face_id': [obj_id],
                'first_seen': today_now,
                'last_seen': [today_now],
                'confidence': [confidence],
                'difference': [difference],
                'seen_frames': [frame_number],
                'image': [image],
                'seen_count': 1
                })

            # actualiza las variables globales
            save_found_faces(camera_id, found_faces)
            update_known_faces_indexes(camera_id, obj_id)
            return True

            # ID already in the list of IDs
        else:
            found_faces = get_found_faces(camera_id)
            i = 0
            for found_face in found_faces:
                # same id
                if found_face['face_id'][-1] == obj_id:
                    today_now = datetime.now()

                    # edicion si se cumple tiempo
                    if today_now - found_faces[i]['last_seen'][-1] > timedelta(seconds=delta):
                        #if confidence - found_face['confidence'][-1] > 0.0038: 
                        if confidence - found_face['confidence'][0] > 0.0038: 
                            img_encoding, img_metadata = biblio.encode_face_image(image, name, camera_id, confidence, False)

                            if img_encoding is not None:
                                metadata, best_index, difference = biblio.lookup_known_face(img_encoding, known_face_encodings, known_face_metadata)

                                if best_index is not None:
                                    found_faces[i]['last_seen'].append(today_now)
                                    found_faces[i]['seen_count'] += 1
                                    found_faces[i]['seen_frames'].append(frame_number)

                                    if found_face['difference'][-1] < difference:
                                        print('Sujeto2.1 {}, encontrado en frame {} con id: {} ,confidence: n{}/o{}, distance: n{}/o{}'.format(metadata['name'], frame_number, obj_id, confidence, found_face['confidence'][-1], difference, found_faces[i]['difference'][-1]))
                                        found_faces[i]['difference'].append(difference)
                                        found_faces[i]['confidence'].append(confidence)
                                        found_faces[i]['image'].append(image)

                                    # actualiza las variables globales
                                    print('Sujeto2.2 {}, encontrado en frame {} con id: {} ,confidence: n{}/o{}, distance: n{}/o{}'.format(metadata['name'], frame_number, obj_id, confidence, found_face['confidence'][-1], difference, found_faces[i]['difference'][-1]))
                                    save_found_faces(camera_id, found_faces)
                                    update_known_faces_indexes(camera_id, obj_id)
                                    return True

                    # edicion si la confidencia es major y eso lleva a una distancia menor
                    if confidence - found_face['confidence'][-1] > 0.0038: 
                        img_encoding, img_metadata = biblio.encode_face_image(image, name, camera_id, confidence, False)

                        if img_encoding is None:
                            return False

                        metadata, best_index, difference = biblio.lookup_known_face(img_encoding, known_face_encodings, known_face_metadata)

                        if best_index is not None:
                            found_faces[i]['last_seen'].append(today_now)
                            found_faces[i]['seen_frames'].append(frame_number)

                            if difference < found_face['difference'][-1]:
                                print('Sujeto2.3 {}, encontrado en frame {} con id: {} ,confidence: n{}/o{}, distance: n{}/o{}'.format(metadata['name'], frame_number, obj_id, confidence, found_face['confidence'][-1], difference, found_faces[i]['difference'][-1]))
                                found_faces[i]['difference'].append(difference)
                                found_faces[i]['confidence'].append(confidence)
                                found_faces[i]['image'].append(image)

                            # actualiza las variables globales
                            print('Sujeto2.4 {}, encontrado en frame {} con id: {} ,confidence: n{}/o{}, distance: n{}/o{}'.format(metadata['name'], frame_number, obj_id, confidence, found_face['confidence'][-1], difference, found_faces[i]['difference'][-1]))
                            save_found_faces(camera_id, found_faces)
                            update_known_faces_indexes(camera_id, obj_id)
                            return True
                i += 1
            return False
    return False


def tiler_src_pad_buffer_probe(pad, info, u_data):
    global fake_frame_number

    num_rects = 0
    gst_buffer = info.get_buffer()

    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
        
    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    
    global total_visitors
    save_image = False
    l_frame = batch_meta.frame_meta_list

    current_pad_index = pyds.NvDsFrameMeta.cast(l_frame.data).pad_index
    camera_id = get_camera_id(current_pad_index)
    program_action = get_action(camera_id)
    delta = get_delta(camera_id)
    default_similarity = get_similarity(camera_id)

    total_visitors, known_face_metadata, known_face_encodings = get_known_faces_db(camera_id)
    tracking_absence_dict = get_tracking_absence_dict(camera_id)
    id_set = set()

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        fake_frame_number += 1
        frame_number = frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
       
        obj_counter = {
        PGIE_CLASS_ID_FACE:0,
        PGIE_CLASS_ID_PLATE:0,
        PGIE_CLASS_ID_MAKE:0,
        PGIE_CLASS_ID_MODEL:0
        }

        contador = 0
        while l_obj is not None:
            contador += 1
            try: 
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            # Periodically check for objects with borderline confidence value that may be false positive detections.
            # If such detections are found, annoate the frame with bboxes and confidence value.

            # Save the annotated frame to file.
            if obj_meta.class_id == 0 and obj_meta.confidence > 0.81:
                # Getting Image data using nvbufsurface
                # the input should be address of buffer and batch_id
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frame_image = crop_and_get_faces_locations(n_frame, obj_meta, obj_meta.confidence)

                if frame_image.size > 0:
                    name = None
                    id_set.add(obj_meta.object_id)
                    known_faces_indexes = get_known_faces_indexes(camera_id)
                    if classify_to_known_and_unknown(camera_id, frame_image, obj_meta.object_id, name, program_action, obj_meta.confidence, fake_frame_number, delta, default_similarity, known_faces_indexes, known_face_metadata, known_face_encodings):
                        save_image = True
                        #cv2.imwrite('/tmp/found_elements/found_multiple_' + str(fake_frame_number) + ".jpg", frame_image)
            try: 
                l_obj = l_obj.next
            except StopIteration:
                break

        #print("Frame Number=", frame_number, "Number of Objects=",num_rects,"Face_count=",obj_counter[PGIE_CLASS_ID_FACE],"Person_count=",obj_counter[PGIE_CLASS_ID_PERSON])
        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        saved_count["stream_"+str(frame_meta.pad_index)] += 1        

        try:
            l_frame=l_frame.next
        except StopIteration:
            break

        if id_set and known_faces_indexes:
            missing_ids = [id_item for id_item in known_faces_indexes if id_item not in id_set ]
            for item in missing_ids:
                if item not in tracking_absence_dict:
                    tracking_absence_dict.update({item: 1})
                else:
                    tracking_absence_dict[item] += 1

    if save_image:
        if program_action == action_types['read']:
            write_to_db(known_face_metadata, known_face_encodings, get_output_db_name(camera_id))
        else:
            target_metadata = get_found_faces(camera_id)
            empty_encodings = []

            for _x_ in range(len(target_metadata)):
                empty_encodings.append('')

            write_to_db(target_metadata, empty_encodings, get_output_db_name(camera_id))

        if id_set and known_faces_indexes:
            known_faces_indexes, tracking_absence_dict = biblio.cleanup_tracking_list(known_faces_indexes, tracking_absence_dict, 80)
            set_tracking_absence_dict(camera_id, tracking_absence_dict)
            set_known_faces_indexes(camera_id, known_faces_indexes)

    return Gst.PadProbeReturn.OK


def write_to_db(face_metadata, face_encodings, output_db_name):
    biblio.write_to_pickle(face_encodings, face_metadata, output_db_name)


def draw_bounding_boxes(image, obj_meta, confidence):
    #confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    #obj_name = pgie_classes_str[obj_meta.class_id]
    #image=cv2.rectangle(image,(left,top),(left+width,top+height),(0,0,255,0),2)
    #image=cv2.line(image, (left,top),(left+width,top+height), (0,255,0), 9)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    #image=cv2.putText(image,obj_name+',C='+str(confidence),(left-5,top-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255,0),2)
    #image = cv2.putText(image, obj_name, (left-5,top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255,0), 2)
    crop_image = image[top-20:top+height+20, left-20:left+width+20]
    return crop_image

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)   
    if(is_aarch64() and name.find("nvv4l2decoder") != -1):
        print("Seting bufapi_version\n")
        Object.set_property("bufapi-version",True)

def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN] <folder to save frames>\n" % args[0])
        sys.exit(1)
    print("Argumentos :",args)

    for i in range(0,len(args)-2):
        fps_streams["stream{0}".format(i)]=GETFPS(i)
    number_sources=len(args)-2
    print("Numero de fuentes :", number_sources)

    global folder_name
    folder_name=args[-1]
    print(folder_name)
    if path.exists(folder_name):
        sys.stderr.write("The output folder %s already exists. Please remove it first.\n" % folder_name)
        sys.exit(1)
    else:
        os.mkdir(folder_name)
        print("Frames will be saved in ", folder_name)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)


    #Emulate reading the information from the server
    '''
    En este punto asumimos que ya se cargaron la funente es decir video o rtsp etc y los valorres se cargaron
    en un dictionario global o las variables ya se cargaron en dictionarios que corresponden a las fuentes con
    sus respectivas configuraciones.

    En este caso no hay diferentes solo leer y buscar y comparar se puede hacer directamente llamando a las
    funcciones en la biblioteca

    De la misma forma los rostros que se quieran cargar puedes cargarse por configuracion indicando explicitamente
    los nombres de los archivos
    '''
    #scfg = biblio.get_server_info()
    #print(scfg)
    #quit()
    camera_id = 'FA:KE:MA:C:AD:DR:ES:S9'
    set_action(camera_id, 'read')
    set_action(camera_id, 'find')
    action = get_action(camera_id)
    pwd = os.getcwd()

    if action == action_types['read']:
        output_db_name = pwd + '/data/video_encoded_faces/test_video_default.data'
        total, encodings, metadata = 0, [] , []
    elif action == action_types['find']:
        output_db_name = pwd + '/data/found_faces/found_faces_db.dat'
        known_faces_db_name = pwd + '/data/encoded_known_faces/knownFaces.dat'

        if com.file_exists(known_faces_db_name):
            set_known_faces_db_name(camera_id, known_faces_db_name)
            total, encodings, metadata = biblio.read_pickle(known_faces_db_name, False)
        else:
            com.log_error('Unable to open {}'.format(known_faces_db_name))

    set_known_faces_db(camera_id, total, encodings, metadata)
    set_output_db_name(camera_id, output_db_name)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        os.mkdir(folder_name+"/stream_"+str(i))
        frame_count["stream_"+str(i)]=0
        saved_count["stream_"+str(i)]=0
        print("Creating source_bin ",i," \n ")
        uri_name=args[i+1]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    
    # Creation of tracking to follow up the model face
    # April 21th
    # ERM
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    
    
    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    if(is_aarch64()):
        print("Creating transform \n ")
        transform=Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    print("Creating EGLSink \n")
    # edgar: cambio esta linea para no desplegar video - 
    #sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink = Gst.ElementFactory.make("fakesink", "fakesink")

    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    # Camaras meraki 720p
    #streammux.set_property('width', 1920)
    streammux.set_property('width', 1280)
    #streammux.set_property('height', 1080)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    print('CURRENT_DIR', CURRENT_DIR)
    pgie.set_property('config-file-path',CURRENT_DIR + "/configs/pgie_config_facenet.txt")
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
        pgie.set_property("batch-size",number_sources)


    # Set properties of tracker
    # April 21th
    # ERM

    config = configparser.ConfigParser()
    config.read('configs/tracker_config.txt')
    config.sections()
    
    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        elif key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        elif key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        elif key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        elif key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        elif key == 'enable-batch-process':
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)


    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    sink.set_property("sync", 0)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")

    # Add tracker in pipeline
    # April 21th
    # ERM

    pipeline.add(pgie)
    pipeline.add(tracker)     # Tracker
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")

    streammux.link(pgie)
    pgie.link(tracker)        # se añade para tracker
    # pgie.link(nvvidconv1)     se modifica
    tracker.link(nvvidconv1)  # se añade para ligar tracker con los demas elementos
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    #tiler_sink_pad=tiler.get_static_pad("sink")
    #if not tiler_sink_pad:
    #    sys.stderr.write(" Unable to get src pad \n")
    #else:
    #    tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
    
    tiler_src_pad=tiler.get_static_pad("src")
    if not tiler_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args[:-1]):
        if (i != 0):
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
