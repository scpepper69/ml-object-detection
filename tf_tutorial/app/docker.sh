docker run -d -it --name od_sample -v "$PWD/../learning:/models/" -p 9005:9005 sleepsonthefloor/graphpipe-tf:cpu --model=/models/frozen_inference_graph.pb --listen=0.0.0.0:9005 

