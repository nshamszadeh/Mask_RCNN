import os
import sys
from visualize_cv2 import model, display_mask, class_names, cv2
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', default='', type=str,
                    help='The filename of image to be completed.')
args = parser.parse_args()
capture = cv2.VideoCapture(args.video)
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
codec = cv2.VideoWriter_fourcc(*'DIVX')
maskout = cv2.VideoWriter('mask.avi', codec, 30.0, size)

frameCount = 0
while(capture.isOpened()):
    ret, frame = capture.read()
    frameCount += 1
    if ret:
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_mask(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        frame = cv2.bitwise_not(frame)
        maskout.write(frame)
        print('frame ', frameCount, ' written')
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
maskout.release()
cv2.destroyAllWindows()
