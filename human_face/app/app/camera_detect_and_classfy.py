import numpy as np
import os
import cv2
import tensorflow as tf

model_file = "../../learning/human_face_detection.pb"

# Keras Image Classification
from tensorflow.keras.models import load_model

keras_file = '../../learning/02_RESNET2_10_frozen_graph.h5'
labels1 = np.array([
        'rx-178',
        'msz-006',
        'rx-93',
        'ms-06'])
classify_model = load_model(keras_file)
classify_model.summary()


# Input Definition
detection_graph = tf.Graph()
with detection_graph.as_default():
    with open(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
sess = tf.Session(graph=detection_graph)

color=(255, 255, 0)
cap = cv2.VideoCapture(0)
cam_height,cam_width = 600,800
height,width = 64,64

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    #img_path = sys.argv[1]
    img_path = path
    img = image.load_img(img_path, target_size=(height, width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 6
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    print(target_layer)

    x = input_model.layers[-1].output
    print(x)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(x)
    model = Model(input_model.layers[0].input, x)

    loss = K.sum(model.layers[-1].output)

    print(layer_name)
    for l in model.layers:
        if l.name == layer_name:
            print(l.name)
            tmp_layer = l
    
    print(tmp_layer)
            
            
    #conv_output = [l for l in model.layers[0].layers if l.name is layer_name][0].output
#    conv_output = [l for l in model.layers if l.name is layer_name][0].output
    conv_output = tmp_layer.output
#    print(conv_output)

    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (height, width))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def draw_box(img, box, color, score):
    x, y, w, h = box
    label = 'face '+str(int(score*100))+'%'  #self.name
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.5, 1)
    cv2.rectangle(img, (x, y), (x + label_size[0], y + label_size[1] + base_line), color, cv2.FILLED)
    cv2.putText(img, label, (x, y + label_size[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0))

def main():
    x, y, w, h = 0, 0, 0, 0

    while(True):
        ret, frame = cap.read()

        image_np = cv2.resize(frame,(cam_width,cam_height))
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Input Definition
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualize Objects
        num_persons=0
        for i in range(boxes[0].shape[0]):
            if scores[0][i] >= 0.5:
                num_persons+=1

                im_height, im_width, _ = image_np.shape
                ymin, xmin, ymax, xmax = tuple(boxes[0][i].tolist())
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)

                x, y, w, h = int(left), int(top), int(right - left), int(bottom - top)
#                draw_box(image_np, (x, y, w, h), color, scores[0][i])

#        cv2.putText(image_np, "There are " + str(num_persons) + " persons.", (0, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
#        cv2.imshow("camera window", image_np)

        if x > 0 and y > 0 and w > 0 and h > 0 :
#            print(h,y,w,x)
#            image_trm = image_np[y : y + h, x : x + w]
            y_top = int(y - (h/3))
            y_bottom = int(y + h + (h/3))
            x_left = int(x - (w/3))
            x_right = int(x + w + (w/3))
#            print(int(y_top), int(y_bottom), int(x_left), int(x_right))
            if y_top > 0 and x_left > 0 and y_bottom < cam_height and x_right < cam_width :
                image_trm = image_np[y_top : y_bottom, x_left : x_right]
                image_trm_resize = cv2.resize(image_trm, dsize=(600,800))

                cv2.imshow("trim window", image_trm_resize)

        if cv2.waitKey(1) == 27:    
            break
        elif cv2.waitKey(1) == ord('p'):
            print("pressed p")

            ary = np.zeros([1, 64, 64, 3], dtype=np.int)
            ary[0] = cv2.resize(image_trm, dsize=(64,64))
            print(ary.shape)
            answer = classify_model.predict(ary,verbose=1)
            print("ラベル#1の確からしさ(%)："+str(np.round(answer[0],decimals=2)*100))
            print("推論結果："+str(labels1[answer[0].argmax()]))

            # テスト対象イメージを準備
#            preprocessed_input = np.zeros([1, height, width, color], dtype=np.int)
#            preprocessed_input[0] = test_img

            predictions = classify_model.predict(ary)[0]
            predicted_class = np.argmax(predictions)
            print(predicted_class)

            # 畳み込み最終層の指定
            cnn_out_layer = "add_8"
 
            # Grad Cam
            cam, heatmap = grad_cam(classify_model, ary, predicted_class, cnn_out_layer)
#            print(cam.astype(np.int))
            print(cam.shape)
#            cv2.imshow("heatmap",cam.astype(np.int))
            cv2.imshow("heatmap",cv2.resize(cam, dsize=(600,800)))

            os.system('PAUSE')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

