docker run -d -it --name od_sample -v "$PWD/../learning:/models/" -p 9021:9021 sleepsonthefloor/graphpipe-tf:cpu --model=/models/frozen_inference_graph.pb --listen=0.0.0.0:9021 

