docker run -d -it --name gface_tfs --rm -p 8501:8501 -p 8500:8500 --mount type=bind,source=/mnt/d/20.programs/github/ml-object-detection/gface/learning/models,target=/models/gface_tfs -e MODEL_NAME=gface_tfs -t tensorflow/serving