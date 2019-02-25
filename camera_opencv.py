import cv2
import time
from base_camera import BaseCamera
from object_detector_lite import ObjectDetectorLite


class Camera(BaseCamera):
    @staticmethod
    def frames():
        detector = ObjectDetectorLite()
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            ret, frame = camera.read()
            print(ret)
            if ret != True:
                time.sleep(2)
                continue
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
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()