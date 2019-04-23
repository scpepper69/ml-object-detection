function pwd_as_linux {
  "/$((pwd).Drive.Name.ToLowerInvariant())/$((pwd).Path.Replace('\', '/').Substring(3))"
}
docker run -d -it --name tf_tutorial --rm -v "$(pwd_as_linux)/../learning:/models/" -p 9005:9005 sleepsonthefloor/graphpipe-tf:cpu --model=/models/frozen_inference_graph.pb --listen=0.0.0.0:9005 
