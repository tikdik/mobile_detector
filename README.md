## Mobile detector for embedded platforms

`master` branch for Raspberry Pi, `jetson-tx2` brach for NVIDIA Jetson TX2.

Use `detect.py` for object detection on image.

Use `detection_stream.py` for detection from Raspberry Pi Camera.

Use `jetson_stream.py` for detection from Jetson Camera.

Use `app.py` for detect raspberry pi usb camera and use flask as video stream sever on http://ip:5000/video_feed

`https://drive.google.com/file/d/1_DSRMQB6oTaPifqAeDz1MIHVmfIaEB4g/view?usp=sharing` - *.pb files with ssdlite model graph.