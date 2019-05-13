function pwd_as_linux {
  "/$((pwd).Drive.Name.ToLowerInvariant())/$((pwd).Path.Replace('\', '/').Substring(3))"
}
docker run -d -it --name gundam --rm -v "$(pwd_as_linux)/../learning:/models/" -p 9006:9006 sleepsonthefloor/graphpipe-tf:cpu1.11.0 --model=/models/frozen_inference_graph.pb --listen=0.0.0.0:9006 
