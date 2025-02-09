import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter


# iPhone Kamera URL'si
stream_url = "http://192.168.1.3:4747/video"
# TFLite modelini yükle
# TFLite modelini yükle
interpreter =tflite.Interpreter(model_path="yolov8n_float16.tflite")
interpreter.allocate_tensors()