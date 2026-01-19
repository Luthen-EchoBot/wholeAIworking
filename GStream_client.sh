#!/usr/bin/env bash

# This should be launched on the pi3 with DISPLAY=:0

gst-launch-1.0 udpsrc port=5000 buffer-size=5000000 ! tsdemux ! h264parse ! avdec_h264 ! videoconvert ! glimagesink sync=false
