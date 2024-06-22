import cv2
import argparse
import time
from threading import Thread
from compreface import CompreFace
from compreface.service import RecognitionService
from datetime import datetime
import datetime
import json
import pandas as pd
import csv

class FaceRecognition:
	def __init__(self, api_key, host, port):
		self.active = True
		self.results = []
		self.capture = cv2.VideoCapture(1)
		self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
		compre_face: CompreFace = CompreFace(host, port, {
			"limit": 0,
			"det_prob_threshold": 0.8,
			"prediction_count": 1,
			"face_plugins": "age,gender",
			"status": False
		})
		self.recognition: RecognitionService = compre_face.init_face_recognition(api_key)
		self.FPS = 1/30
		self.thread = Thread(target=self.show_frame, args=())
		self.thread.daemon = True
		self.thread.start()
		self.json_file = "recognition_log.json"
		self.csv_file = "recognition_log.csv"
		self.offset_width = 200
		self.offset_high = 200
		self.written_id = []
		self.checking_state = False
		self.list_data = []
		self.infor_dict = {}
		self.i = 0

	def show_frame(self):
		print("Started")
		while self.capture.isOpened():
			(status, frame_raw) = self.capture.read()
			self.frame = cv2.flip(frame_raw, 1)

			if self.results:
				results = self.results
				for result in results:
					box = result.get('box')
					subjects = result.get('subjects')
					if box:
						cv2.rectangle(img=self.frame, pt1=(box['x_min'], box['y_min']),
							pt2=(box['x_min'] + self.offset_width, box['y_min'] + self.offset_high), color=(0, 255, 0), thickness=1)
					if subjects:
						subjects = sorted(subjects, key=lambda k: k['similarity'], reverse=True)
						name = f"Name: {subjects[0]['subject']}"
						acc = f"Accuracy: {subjects[0]['similarity']}"
						ID = f"ID: {subjects[0]['subject']}"
						accuracy = subjects[0]['similarity']
						person = subjects[0]['subject']
						ident = subjects[0]['subject']
						current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
						cv2.putText(self.frame, name, (box['x_max'], box['y_min'] + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
						cv2.putText(self.frame, acc, (box['x_max'], box['y_min'] + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
						cv2.putText(self.frame, ID, (box['x_max'], box['y_min'] + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
						data = [ident, person, accuracy, current_time]
						self.list_data.append(data)
						# print(self.list_data)
						for item in self.list_data:
							key = item[0]
							value = item[2:4]

							if key not in self.infor_dict:
								self.infor_dict[key] = [value]
							else:
								if self.infor_dict[key]:
									first_item = self.infor_dict[key][0]
									first_ts = datetime.datetime.strptime(first_item[1], '%Y-%m-%d %H:%M:%S')
									last_item = self.infor_dict[key][-1]
									last_ts = datetime.datetime.strptime(last_item[1], '%Y-%m-%d %H:%M:%S')
									last_acc = self.infor_dict[key][0][0]

									if ((datetime.datetime.strptime(item[3], '%Y-%m-%d %H:%M:%S') - last_ts).total_seconds() > 60):
										self.infor_dict[key] += [value]
								else:
									self.infor_dict[key] += [value]
						print(self.infor_dict)

						with open('data.csv', mode='w', newline = '') as f:
								writer = csv.writer(f)
								writer.writerow(['STT' ,'ID', 'Name', 'Accuracy', 'Timestamp'])
								i = 1
								for ID, accuracy in self.infor_dict.items():
									for acc, timestamp in accuracy:
										if acc > 0.95:
											writer.writerow([i ,ID, ID, acc, timestamp])
											i += 1
					else:
						subject = f"No known faces"
						cv2.putText(self.frame, subject, (box['x_max'], box['y_min'] + 75),
									cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
			cv2.imshow('Test Webcam', self.frame)
			time.sleep(self.FPS)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				self.capture.release()
				cv2.destroyAllWindows()
				self.active=False

	def is_active(self):
		return self.active

	def update(self):
		if not hasattr(self, 'frame'):
			return
		_, im_buf_arr = cv2.imencode(".jpg", self.frame)
		byte_im = im_buf_arr.tobytes()
		data = self.recognition.recognize(byte_im)
		self.results = data.get('result')

if __name__ == '__main__':
    face_recognition = FaceRecognition("1faca120-f623-470b-bebb-fc7903890ea4", "http://localhost", "8000")
    while face_recognition.is_active():
        face_recognition.update()