#!/bin/bash
rm -f data/video_encoded_faces/*
rm -rf /tmp/res; python3 car_plates.py file:///tmp/HD_CCTV_Camera.mp4 /tmp/res
ssh edgar@192.168.130.5 rm -rf /tmp/res/; scp -r /tmp/res/stream_0 edgar@192.168.130.5:/tmp
