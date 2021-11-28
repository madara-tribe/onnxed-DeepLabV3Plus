import sys, os
import time
import numpy as np
import torch
import onnxruntime
import cv2
from mask2segmap import give_color_to_seg_img

NUM_CLS = 5
crop_size=35
W = int(1216/4)
H = int(1936/4)

def gamma(img, gamma = 3.0):
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')
    for i in range(256):
         gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
    return cv2.LUT(img, gamma_cvt)
    
def inference(onnx_file_name, image_path):
    # Input
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (H, W), cv2.INTER_NEAREST)
    img = gamma(img, gamma = 3.0)
    img_in = np.transpose(img[:W-crop_size,:], (2, 0, 1))
    img_in = np.expand_dims(img_in, axis=0)/255
    print("Shape of the network input: ", img_in.shape, img_in.min(), img_in.max())

    # onnx runtime
    ort_session = onnxruntime.InferenceSession(onnx_file_name)
    IMAGE_HEIGHT = ort_session.get_inputs()[0].shape[2]
    IMAGE_WIDTH = ort_session.get_inputs()[0].shape[3]
    input_name = ort_session.get_inputs()[0].name
    print("The model expects input shape: ", ort_session.get_inputs()[0].shape)
    
    # prediction
    print('start calculation')
    start_time = time.time()
    outputs = ort_session.run(None, {input_name: img_in.astype(np.float32)})[0]
    preds = np.argmax(outputs, axis=1).reshape(IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.uint8)
    segimg = give_color_to_seg_img(preds, n_classes=NUM_CLS)
    cv2.imwrite('pred_segmap.png', (segimg*255).astype(np.uint8))
    print("Inference Latency (ms) until saved is", (time.time() - start_time)*1000, "[ms]")
    segimg = (segimg*255).astype(np.uint8)
    print('preds.shape', preds.shape, np.unique(preds))
    print('segimg.shape', segimg.shape, np.unique(segimg))


if __name__=='__main__':
    image_path = str(sys.argv[1])
    onnx_file_name = str(sys.argv[2])
    inference(onnx_file_name, image_path)



