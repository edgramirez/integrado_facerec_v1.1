#!/bin/bash
rm -f /tmp/data/video_encoded_faces/*
#rm -rf /tmp/stream_0; python3 car_plates.py file:///tmp/obama_biden.mp4    /tmp/stream_0
rm -rf /tmp/stream_0; python3 car_plates.py file:///tmp/HD_CCTV_Camera.mp4 /tmp/stream_0
ssh edgar@192.168.130.5 rm -rf /tmp/stream_0; scp -r /tmp/stream_0 edgar@192.168.130.5:/tmp
