function pwd_as_linux {
  "/$((pwd).Drive.Name.ToLowerInvariant())/$((pwd).Path.Replace('\', '/').Substring(3))"
}
docker run -d -it --name tf_tutorial --rm -v "$(pwd_as_linux)/../learning:/models/" -p 9021:9021 sleepsonthefloor/graphpipe-tf:cpu --model=/models/tf_tutorial_detection.pb --listen=0.0.0.0:9021 
