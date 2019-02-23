import numpy as np
import tensorflow as tf
import cv2
from os import path
import time

from utils import label_map_util
from object_detector import ObjectDetector


basepath = path.dirname(__file__)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = path.join(basepath, 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

class ObjectDetectorLite:
    def __init__(self, model_path='detect.tflite'):
        """
            Builds Tensorflow graph, load model and labels
        """

        # Load lebel_map
        self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

        # Define lite graph and Load Tensorflow Lite model into memory
        self.interpreter = tf.contrib.lite.Interpreter(
            model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image, threshold=0.1):
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """

        # Resize and normalize image for network input
        frame = cv2.resize(image, (300, 300))
        frame = np.expand_dims(frame, axis=0)
        frame = (2.0 / 255.0) * frame - 1.0
        frame = frame.astype('float32')

        # run model
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        num = self.interpreter.get_tensor(
            self.output_details[3]['index'])

        # Find detected boxes coordinates
        return self._boxes_coordinates(image,
                            np.squeeze(boxes[0]),
                            np.squeeze(classes[0]+1).astype(np.int32),
                            np.squeeze(scores[0]),
                            min_score_thresh=threshold)

    def _load_label(self, path, num_c, use_disp_name=True):
        """
            Loads labels
        """
        label_map = label_map_util.load_labelmap(path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_c,
                                                                    use_display_name=use_disp_name)
        self.category_index = label_map_util.create_category_index(categories)

    def _boxes_coordinates(self,
                            image,
                            boxes,
                            classes,
                            scores,
                            max_boxes_to_draw=20,
                            min_score_thresh=.5):
        """
          This function groups boxes that correspond to the same location
          and creates a display string for each detection and overlays these
          on the image.

          Args:
            image: uint8 numpy array with shape (img_height, img_width, 3)
            boxes: a numpy array of shape [N, 4]
            classes: a numpy array of shape [N]
            scores: a numpy array of shape [N] or None.  If scores=None, then
              this function assumes that the boxes to be plotted are groundtruth
              boxes and plot all boxes as black with no classes or scores.
            category_index: a dict containing category dictionaries (each holding
              category index `id` and category name `name`) keyed by category indices.
            use_normalized_coordinates: whether boxes is to be interpreted as
              normalized coordinates or not.
            max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
              all boxes.
            min_score_thresh: minimum score threshold for a box to be visualized
        """

        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        number_boxes = min(max_boxes_to_draw, boxes.shape[0])
        person_boxes = []
        # person_labels = []
        for i in range(number_boxes):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box

                im_height, im_width, _ = image.shape
                left, right, top, bottom = [int(z) for z in (xmin * im_width, xmax * im_width,
                                                             ymin * im_height, ymax * im_height)]

                person_boxes.append([(left, top), (right, bottom), scores[i],
                                     self.category_index[classes[i]]['name']])
        return person_boxes

    def close(self):
        pass


if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    detector = ObjectDetectorLite()

    cap=cv2.VideoCapture(0)#'video1.mp4')#文件名及格式
    stop_time = time.time() + 40.0
    while(True):
        #capture frame-by-frame
        ret , frame = cap.read()
        if ret != True or stop_time <= time.time():
            break

        #our operation on the frame come here
        image = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        result = detector.detect(image, 0.4)
        print(result)

        for obj in result:
            print('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                        format(obj[0], obj[1], obj[3], obj[2]))

            cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
            cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                        (obj[0][0], obj[0][1] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        #display the resulting frame
        out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cap.release()
    out.release()

    # image = cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB)

    # result = detector.detect(image, 0.4)
    # print(result)

    # for obj in result:
    #     print('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
    #                 format(obj[0], obj[1], obj[3], obj[2]))

    #     cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
    #     cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
    #                 (obj[0][0], obj[0][1] - 5),
    #                 cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # cv2.imwrite('r1.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # detector.close()