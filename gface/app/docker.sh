docker run -d -it --name gface_detect --rm -v "$PWD/../learning:/models/" -p 9023:9023 scpepper/graphpipe-tf:cpu1.13.1 --model=/models/gface_detection.pb --listen=0.0.0.0:9023
