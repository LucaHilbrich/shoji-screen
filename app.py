from torchvision import models
from PIL import Image
import cv2
import numpy as np
import collections
from ffpyplayer.player import MediaPlayer

import utils as ut

# --- PARAMETERS ------------------#
VIDEO_PATH = './input/tree2.mp4'
AUDIO_PATH = './input/tree2.mp4'
MODEL_INPUT_SIZE = 60
SMOOTHING = True
SMOOTHING_KERNEL_SIZE = 35
TEMPORAL_SMOOTHING = True
# ---------------------------------#

if __name__ == "__main__":

    kernel = np.ones((SMOOTHING_KERNEL_SIZE, SMOOTHING_KERNEL_SIZE), np.float32) / (SMOOTHING_KERNEL_SIZE ** 2)

    volume_queue = collections.deque(np.zeros(8), maxlen=8)
    frame_queue = []

    # fastest model: fcn_resnet50
    dlab = models.segmentation.fcn_resnet50(pretrained=True).eval()

    bg = cv2.VideoCapture(VIDEO_PATH)
    bg.set(cv2.CAP_PROP_POS_FRAMES, 300)
    _, bg_static = bg.read()
    height, width, _ = bg_static.shape
    bg.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    bg.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    bg_static = cv2.resize(bg_static, (width, height))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    player = MediaPlayer(AUDIO_PATH)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        ret_bg, frame_bg = bg.read() 
        audio_frame, val = player.get_frame()

        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break

        if ret_bg:
            mask = ut.modify(dlab, Image.fromarray(cv2.flip(frame, 1)), MODEL_INPUT_SIZE)

            if 16 in mask:
                volume_queue.appendleft(0.5)
                volume_queue.pop()
            else:
                volume_queue.appendleft(0)
                volume_queue.pop()
            # print(np.mean(volume_queue))

            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            if TEMPORAL_SMOOTHING:
                frame_queue.append(mask)
                if len(frame_queue) > 5:
                    frame_queue.pop(0)
                mask = np.mean(frame_queue, axis=0)

            mask = np.where(mask==15, 1, 0).astype('uint8')
            if SMOOTHING:
                mask = cv2.filter2D(mask, -1, kernel)
            mask_inv = 1 - mask
            out = frame_bg * cv2.merge((mask, mask, mask)) + bg_static * cv2.merge((mask_inv, mask_inv, mask_inv))
            cv2.imshow('frame', out)

            if val != 'eof' and audio_frame is not None:
                #audio
                player.set_volume(np.mean(volume_queue))
                img, t = audio_frame
            
        else:
            bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()