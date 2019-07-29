docker run -d -it --name human_face --rm -v "$PWD/../learning:/models/" -p 9022:9022 scpepper/graphpipe-tf:cpu1.13.1 --model=/models/human_face_detection.pb --listen=0.0.0.0:9022
