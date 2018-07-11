from kivy.core.window import Window
Window.size = (350, 600)
from kivy.config import Config
Config.set('graphics','resizable',0)

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
import cv2
from kivy.core.window import Window
import subprocess,os
from os.path import join as pjoin
import numpy as np

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import argparse
from kivy.uix.progressbar import ProgressBar


name = []
objects = []

kv = '''
main:
	BoxLayout:
		orientation: 'vertical'
		padding: root.width * 0.05, root.height * .05
        ProgressBar:
            id: bar
            max: 2000 
            value: 1600
		BoxLayout:
			size_hint: [1,.3]
			orientation: 'horizontal'
			Image:
				source: 'image.jpg'
			Label:
				text: "Video Analysis"
				bold: True


				Button:
					text: "Analyse"
					bold: True
					pos: 0.5, 0
					background_normal: 'red.jpg'
					on_press: root.Cyclop()

'''

class main(BoxLayout):
    def convert(self,):
        source = self.ids.source.text
        dest  = self.ids.dest.text
        ext = self.ids.ext.text
        destination_filename = dest+ext
        

        try:
            image = cv2.imread(source)
            cv2.imwrite(destination_filename, image)
            popup = Popup(title='Done', content=Label(text='Image Converted'),size_hint=(.5, .2))
            popup.open()
        except:
            popup = Popup(title='Error', content=Label(text='Error converting'), size_hint=(.5, .2))
            popup.open()

    def Cyclop(self):
	    
        if tf.__version__ < '1.4.0':
            raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
  
        #Ceci est nécessaire pour afficher les images.
        
        sys.path.append("..")

        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as vis_util
        
        #for file in os.listdir(paths):
            #filename = os.fsdecode(file)
            #if filename.endswith(".mp4") or filename.endswith(".py"): 
                #name = os.path.join(paths, filename)
                #names.append(name)
                
                #continue
            #else:
                #continue

        #file, namefile = os.path.split(file)
        #video = file+'\\'+namefile+ "'"

        #ap = argparse.ArgumentParser()
        #ap.add_argument("-v", "--name", required=True,
	 #   help="path to input video file")
        #args = vars(ap.parse_args())
        #print(name)
        
        #fish = 'E:\\Tesearch\\Future.mp4'
        ##fish = name[0]
        #new = fish.replace('\\','\\\\')
        #new1 = "'"+new+"'"
        
        #print(fish)

        cap=cv2.VideoCapture(0)

        MODEL_NAME = 'object_detector_app-master/object_detection/ssd_mobilenet_v1_coco_11_06_2017'
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        PATH_TO_LABELS = os.path.join('object_detection\data', 'mscoco_label_map.pbtxt')
        print('Be Amazed',  )
        print("------------------------------------------ %s",PATH_TO_CKPT)
        NUM_CLASSES = 90


        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
	
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        
	  
	  
        PATH_TO_TEST_IMAGES_DIR = 'image'
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)

        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            #Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            boc = []
            clas = []
            counting = 0
            threshold = 0.5
            ppl = 'person'
            nom = []
            #set1 = set(ppl.split(' '))
            
            while True:
      
	  
              ret,image_np=cap.read()
      
              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(image_np, axis=0)
              # Actual detection.
              (boxes, scores, classes, num) = sess.run(
                 [detection_boxes, detection_scores, detection_classes, num_detections],
                   feed_dict={image_tensor: image_np_expanded})
              # Visualization of the results of a detection.
	
              final_score = np.squeeze(scores)
              count = 0
              for i in range(100):
                if scores is None or final_score[i] > 0.5:
                  count = count + 1
	
              final_class = (classes).astype(np.int32)
              
              #for name1 in final_class :
                #print(name1)
                  
              vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
              cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
              #videowriter = cv2.VideoWriter(image_np,(800,600))
              if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                BREAK
             # plt.figure(figsize=IMAGE_SIZE)
             # plt.imshow(image_np)
             # plt.show()
              #print(boxes[0])
              #len(boxes.shape)
			  #boc.append(boxes.shape)
              b = count + count
              print(scores.shape[0])
              print(len(boxes.shape))
              print(count)
              clas.append(count)
              global c
              
              c = sum(clas)
              print(c)
              pb = ProgressBar(max=2000)
              c = pb.value
              #print([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])		  
              #print(final_score[0])		  
              print('paul')		  
              for index, value in enumerate(classes[0]):
                object_dict = {}
                if scores[0, index] > threshold:
                  object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                                        scores[0, index]
                  objects.append(object_dict)
              if ( (category_index.get(value)).get('name') == ppl):
                counting = counting + 1
              print ('num person: ',counting)				  
              #print (objects)				  
              print ((category_index.get(value)).get('name'))				  
              nom.append((category_index.get(value)).get('name'))				  
              				  
              #print ('denis :', nom)				  
              a = list(set(nom))				  
              print(a)				  
              print('Number of Objects: ',len(a))				  
              				  
        os.system("pause")
    #print(objects)


    def load_image_into_numpy_array(image):
          (im_width, im_height) = image.size
          return np.array(image.getdata()).reshape(
              (im_height, im_width, 3)).astype(np.uint8) 


class imageConvert(App):
	def build(self):
		Window.bind(on_dropfile=self._on_file_drop)

		return Builder.load_string(kv)

	def _on_file_drop(self, window, file_path): 
    
        
		global file
		file = file_path
		
		name.append(file_path.decode('utf-8'))
		print(file_path)
		print(name)
		return 
		
	
imageConvert().run()