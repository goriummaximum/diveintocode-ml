import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

common = {
	'DATA_SRC': '/content/drive/MyDrive/Colab-Notebooks/grad-ass/data/za_traffic_2020/traffic_train',
	'MODEL_SRC': '/content/drive/MyDrive/Colab-Notebooks/grad-ass/model'
}

class BBoxVisualizer:
	def __init__(self, box_color, text_color):
		self.BOX_COLOR = box_color
		self.TEXT_COLOR = text_color

	def visualize_bbox(
		self,
		img,
		bbox,
		class_name,
		thickness=2,
		font=r'/content/SVN-Arial Regular.ttf',
		format='pascal_voc',
		implementation='pil'
    ):
		"""Visualizes a single bounding box on the image"""
		x_min = None
		y_min = None 
		x_max = None
		y_max = None
		if (format == 'coco'):
			x_min, y_min, w, h = bbox
			x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
		
		elif (format == 'pascal_voc'):
			x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
		
		if (implementation == 'cv2'):
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=self.BOX_COLOR, thickness=thickness)
			
			((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)    
			cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), self.BOX_COLOR, -1)
			cv2.putText(
				img,
				text=class_name,
				org=(x_min, y_min - int(0.3 * text_height)),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=0.5, 
				color=None, 
				lineType=cv2.LINE_AA,
			)
		
		elif (implementation == 'pil'):
			if isinstance(img, np.ndarray):
				img = Image.fromarray(img)
			draw = ImageDraw.Draw(img)

			draw.rectangle([(x_min, y_min), (x_max, y_max)], fill=None, outline=self.BOX_COLOR, width=thickness)

			font_text = ImageFont.truetype(font=font, size=20)
			text_width, text_height = font_text.getsize(class_name)
			draw.rectangle([(x_min, y_min - int(1.3 * text_height)) , (x_min + text_width, y_min)], fill=self.BOX_COLOR)
			draw.text((x_min, y_min - int(1.3 * text_height)), class_name, self.TEXT_COLOR, font=font_text)
			img = np.asarray(img)

		return img
	
	def visualize(self, image, bboxes, labels, file_name, format, implementation, plt_show=True):
		img = image.copy()
		for bbox, label in zip(bboxes, labels):
			img = self.visualize_bbox(img=img, bbox=bbox, class_name=label, format=format, implementation=implementation)
		if (plt_show):
			plt.figure(figsize=(12, 12))
			plt.title('{}'.format(file_name))
			plt.axis('off')
			plt.imshow(img)
		
		return img
		
