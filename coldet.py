import argparse

import numpy as np
import webcolors
from cv2 import cv2
from skimage import io


def load_yolo():
	#Load the yolov3 weights and config file with the help of dnn module of openCV.

	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):		
	#Use blobFromImage that accepts image/frame from video or webcam stream, model and output layers as parameters.	

	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	#The list scores is created which stores the confidence corresponding to each object. Can play around with this value.

	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			#print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def closest_colour(requested_colour):
	#webcolors raises an exception if it can't find the match for a requested color.

    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
	#this fix delivers the closest matching name for the requested RGB colour.

    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def detect_color(frame,a,b,c,d):
	"""Function where the color detection takes place"""

	try:
		x,y,w,h = a,b,c,d

		roi = frame[y:y+h,x:x+w]
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
		average = roi.mean(axis=0).mean(axis=0) 
		#apply k-means clustering to create a palette with the most representative colours of the image
		pixels = np.float32(roi.reshape(-1, 3)) 
		n_colors = 5
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
		_, counts = np.unique(labels, return_counts=True)
		dominant = palette[np.argmax(counts)] #dominant colour is the palette colour which occurs most frequently on the quantized image

		#convert into a format suitable for color name indentifier function
		dominant=dominant.astype(int)
		dominant=dominant.tolist()
		dominant=tuple(dominant)
		requested_colour = dominant
		actual_name, closest_name = get_colour_name(requested_colour)
		return closest_name
	except Exception:
		pass


def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	#Draw bounding box and add object labels to it.

	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
			if x>0 and y>0 and w>0 and h > 0:
				closest_name = detect_color(img,x,w,y,h)
				cv2.putText(img, closest_name, (x, y- 15), font, 1,color,1 )

	cv2.imshow("Image", img)


def start_video(video_path):
	#Pipeline the other functions together and read video file frame by frame performing detections on it.

	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	out = cv2.VideoWriter('outpy1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		out.write(frame)

		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()
	


start_video("test.MOV")
