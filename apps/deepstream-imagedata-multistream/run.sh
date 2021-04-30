#!/bin/bash
rm -f /tmp/data/video_encoded_faces/*
rm -f data/video_encoded_faces/*
rm -f data/encoded_known_faces/*
rm -f /tmp/found_elements/*

#./faces.py loadFaces
#cp data/encoded_known_faces/knownFaces.dat /tmp/data/encoded_known_faces/
#ls -lartc data/encoded_known_faces/knownFaces.dat

#rm -rf /tmp/stream_0; python3 car_plates.py file:///tmp/obama_biden.mp4    /tmp/stream_0
rm -rf /tmp/stream_0; python3 car_plates.py file:///tmp/amlo.mp4           /tmp/stream_0
#rm -rf /tmp/stream_0; python3 car_plates.py file:///tmp/HD_CCTV_Camera.mp4 /tmp/stream_0

exit 111
ssh edgar@192.168.130.5 'rm -rf /tmp/stream_0'
ssh edgar@192.168.130.5 'rm -f /tmp/resultados/*'

scp -r /tmp/stream_0         edgar@192.168.130.5:/tmp
scp    /tmp/found_elements/* edgar@192.168.130.5:/tmp/resultados/
