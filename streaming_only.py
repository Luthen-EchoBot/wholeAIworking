
print("Importing openCV...")
import cv2
# print("Importing numpy...")
# import numpy as np
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 1 GStreamer Pipeline: BGR -> I420 -> HW H.264 Encoder -> RTP -> UDP
# gst_out = (
#         "appsrc ! "
#         "video/x-raw, format=BGR ! "
#         "queue ! "
#         "videoconvert ! "
#         "video/x-raw, format=I420 ! "
#         "nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! "  # Hardware encoder
#         "rtph264pay ! "
#         "udpsink host=192.168.1.20 port=5000 sync=false"
#         )

# 2 GStreamer Pipeline: BGR (OpenCV) -> BGRx (CPU) -> NV12 (GPU/NVMM) -> H.264 (HW) -> RTP -> UDP
# gst_out = (
#     "appsrc ! "
#     "video/x-raw, format=BGR ! "
#     "queue ! "
#     "videoconvert ! "
#     "video/x-raw, format=BGRx ! "
#     "nvvideoconvert ! "
#     "video/x-raw(memory:NVMM), format=NV12 ! " 
#     "nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! "
#     "rtph264pay ! "
#     "udpsink host=192.168.1.20 port=5000 sync=false"
# )

# Fallback: Software Encoding (CPU only)
# gst_out = (
#     "appsrc ! "
#     "video/x-raw, format=BGR ! "
#     "videoconvert ! "
#     "x264enc tune=zerolatency bitrate=600 speed-preset=ultrafast ! "
#     "rtph264pay config-interval=1 pt=96 ! "  # "rtph264pay ! "
#     "udpsink host=192.168.1.20 port=5000 sync=false"
# )

# MPEG-TS Container: Robust and self-describing
# gst_out = (
#     "appsrc ! "
#     "video/x-raw, format=BGR ! "
#     "videoconvert ! "
#     "x264enc tune=zerolatency speed-preset=ultrafast ! " 
#     "h264parse ! "             # Fixes stream flags for the muxer
#     "mpegtsmux alignment=7 ! "             # Packages video into a TV-like stream
#     "udpsink host=192.168.1.20 port=5000 sync=false"
# )

# gst_out = (
#     "appsrc ! "
#     "video/x-raw, format=BGR ! "
#     "videoconvert ! "
#     "x264enc tune=zerolatency speed-preset=ultrafast bitrate=20000 ! " 
#     "h264parse ! "             
#     "mpegtsmux alignment=7 ! " 
#     "udpsink host=192.168.1.20 port=5000 sync=false"
# )

# Working, high fps, high bitrate, high delay (~5s)
# gst_out = (
#     "appsrc ! "
#     "video/x-raw, format=BGR ! "
#     "queue ! "
#     "videoconvert ! "
#     "video/x-raw, format=I420 ! "
#     # Software Encoder tuned for High Quality + Low Latency
#     # bitrate=5000 (5Mbps) provides DVD-quality for 640x480
#     # speed-preset=superfast is the sweet spot: good visuals, low CPU usage
#     "x264enc tune=zerolatency bitrate=5000 speed-preset=superfast key-int-max=30 ! "
#     "rtph264pay config-interval=1 pt=96 ! "
#     "udpsink host=192.168.1.20 port=5000 sync=false"
# )
# 
# stream_out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, 30.0, (640, 480))

gst_out = (
    "appsrc ! "
    "video/x-raw, format=BGR ! "
    # 1. Force 4 FPS
    "videorate ! video/x-raw,framerate=4/1 ! "
    "videoconvert ! "
    # 2. DOWNSCALE: Resize to 320x240 (Quarter VGA)
    # This is the biggest speed boost possible.
    "videoscale ! video/x-raw,width=320,height=240 ! "
    "video/x-raw, format=I420 ! "
    # 3. FASTEST ENCODING:
    # ultrafast = lowest CPU usage (lowest latency)
    # bitrate=300 = Tiny packets, practically instant transmission
    "x264enc tune=zerolatency bitrate=300 speed-preset=ultrafast key-int-max=4 ! "
    "queue max-size-buffers=1 leaky=downstream ! "
    "rtph264pay config-interval=1 pt=96 mtu=1400 ! "
    "udpsink host=192.168.1.20 port=5000 sync=false"
)
gst_out = (
    "appsrc ! "
    "video/x-raw, format=BGR ! "
    # 1. Force 4 FPS. This creates "breathing room" for the network.
    "videorate ! video/x-raw,framerate=4/1 ! " 
    "videoconvert ! "
    # "video/x-raw, format=I420 ! "
    "videoscale ! video/x-raw,width=320,height=240, format=I420 ! "
    # 2. Bitrate 800 is plenty for 4fps. 
    # key-int-max=4 ensures a full refresh every 1 second.
    "x264enc tune=zerolatency bitrate=800 speed-preset=superfast key-int-max=4 ! "
    # 3. Aggressive Leaky Queue (Back to size 1)
    # Since bandwidth is low, we won't overflow, so this is safe now.
    "queue max-size-buffers=1 leaky=downstream ! "
    "rtph264pay config-interval=1 pt=96 mtu=1400 ! "
    "udpsink host=192.168.1.20 port=5000 sync=false"
)
stream_out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, 4.0, (640, 480))

if not stream_out.isOpened():
    raise RuntimeError("Failed to open GStreamer pipeline")

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Camera read failed")
        break
    # frame = results[0].plot()
    stream_out.write(frame)

stream_out.release()
cap.release()
