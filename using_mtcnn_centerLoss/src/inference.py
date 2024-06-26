from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import facenet
import cv2
import os
import numpy as np
import math
import pickle
from sklearn.svm import SVC
import sys
import argparse
from using_mtcnn_centerLoss.align_processing import detect


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Loaded classifier file")

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect.create_mtcnn(sess, 'align')
            print("Loaded MTCNN")
            facenet.load_model(FACENET_MODEL_PATH)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            people_detected = set()
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                bounding_boxes, _ = detect.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                faces_found = bounding_boxes.shape[0]
                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faces_found):
                        emb_array = np.zeros((1, embedding_size))
                        bb[i][0] = np.maximum(bb[i][0], 0)
                        bb[i][1] = np.maximum(bb[i][1], 0)
                        bb[i][2] = np.minimum(bb[i][2], frame.shape[1])
                        bb[i][3] = np.minimum(bb[i][3], frame.shape[0])
                        print(bb[i][3], bb[i][2], bb[i][1], bb[i][0])
                        print("Face Detected")
                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(cv2.resize(cropped[i], (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                 interpolation=cv2.INTER_CUBIC))
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                        if best_class_probabilities > 0.9:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                            if best_name not in people_detected:
                                people_detected.add(best_name)
                                cv2.putText(frame, best_name, (bb[i][0], bb[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                        else:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (255, 0, 0), 2)
                            cv2.putText(frame, 'Unknown', (bb[i][0], bb[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)

                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()

