function pwd_as_linux {
  "/$((pwd).Drive.Name.ToLowerInvariant())/$((pwd).Path.Replace('\', '/').Substring(3))"
}
docker run -d -it --name gface_detect --rm -v "$(pwd_as_linux)/../learning:/models/" -p 9023:9023 sleepsonthefloor/graphpipe-tf:cpu1.11.0 --model=/models/frozen_inference_graph.pb --listen=0.0.0.0:9023 
