import time
import bulletin
import numpy as np
import cv2
import scipy.io.wavfile as wav

board = bulletin.Bulletin()
imgs = []
imgs.append(np.rollaxis(cv2.cvtColor(cv2.imread('data/1.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.0, 2, 0))
imgs.append(np.rollaxis(cv2.cvtColor(cv2.imread('data/2.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.0, 2, 0))

pixel_attentions = []
for i in range(16):
    pixel_attentions += [np.clip(i / 16 * np.arange(0, 1, 1 / (16)), 0, 1)]
attention = np.vstack(pixel_attentions)

cap = cv2.VideoCapture('data/video.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video = np.empty((frameCount, 3, frameHeight, frameWidth))
fs, audio = wav.read("data/audio.wav")

fc = 0
ret = True

while (fc < frameCount and ret):
    ret, frame = cap.read()
    video[fc] = np.rollaxis(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0, 2, 0)
    fc += 1

cap.release()

board.CreateImage("image", imgs[0])
board.CreateVideo("video", video, audio=audio)

graph = board.CreateGraph("Graph", ["1st quantity", "2nd quantity"], axis_x="iteration", axis_y="value")
for i in range(20):
    board.create_image_attention("attention", imgs[i % 2], attention)
    graph.AddPoint(i, [i, 2 * i])
    board.Post()
    time.sleep(1)
