import visdom
import numpy as np
from sklearn.manifold import TSNE
import csv
import matplotlib.cm as cm
import imageio
import menpo
import os
import tempfile
import cv2
import math

try:
    from .html_table import table
except:
    from html_table import table

import scipy.io.wavfile as wav
import ffmpeg
from scipy import signal
import warnings

FACE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
              (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # chin
              (17, 18), (18, 19), (19, 20), (20, 21),  # right eyebrow
              (22, 23), (23, 24), (24, 25), (25, 26),  # left eyebrow
              (27, 28), (28, 29), (29, 30),  # nose bridge
              (31, 32), (32, 33), (33, 34), (34, 35),  # nose tip
              (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # right eye
              (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # left eye
              (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
              (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # outer mouth
              (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66),
              (66, 67), (67, 60)]  # inner mouth


# Base class used for type filtering
class Post():
    def __init__(self, types=None):
        self.win = None
        if types is None:
            self.types = {}
        else:
            self.types = types


def swp_extension(file, ext):
    return os.path.splitext(file)[0] + ext


def filify(string):
    filename = string.replace(" ", "_")
    filename = filename.replace(":", "-")
    filename = filename.replace("-_", "-")
    return filename


class Scatter(Post):
    def __init__(self, id, datapoints, labels=None, sequence_coloring=True, t_sne=False, perplexity=10, iterations=2000, filter_name=None,
                 board=None):
        super().__init__(types={"charts"})
        self.id = id
        self.board = board
        self.no_points = datapoints.shape[0]
        self.label_mapping = {}
        self.labels = labels
        self.names = None
        self.filter_name = filter_name

        self.change_labelling(labels)

        self.sequence_coloring = sequence_coloring
        if t_sne:
            TSNE_Mapper = TSNE(n_components=2, perplexity=perplexity, n_iter=iterations)
            self.datapoints = TSNE_Mapper.fit_transform(datapoints)
        else:
            self.datapoints = datapoints

    def change_labelling(self, labels, filter_name=None):
        self.label_mapping = {}
        self.labels = labels
        self.names = None
        self.filter_name = filter_name

        if (labels is None) or (isinstance(labels[0], int) and (1 in labels)):
            return

        no_entries = 1
        for name in labels:
            if name not in self.label_mapping:
                self.label_mapping[name] = no_entries
                no_entries += 1

        self.labels = list(map(self.label_mapping.get, self.labels))
        self.names = list(sorted(self.label_mapping, key=self.label_mapping.__getitem__))

    def post(self):
        if self.board is None:
            return

        win_name = self.id
        if self.filter_name is not None:
            win_name += "Filtered by: " + self.filter_name

        if self.labels is None:
            if self.sequence_coloring:
                colors = 255 * (cm.coolwarm(np.arange(0, 1, step=1.0 / self.no_points))[:, :3])
            self.board.scatter(X=self.datapoints,
                               opts=dict(title=self.id,
                                         markercolor=colors.astype(int),
                                         markersize=5,
                                         ),
                               win=win_name)
        else:
            if self.names is not None:
                self.board.scatter(X=self.datapoints,
                                   Y=self.labels,
                                   opts=dict(title=win_name,
                                             legend=self.names,
                                             markersize=5,
                                             ),
                                   win=win_name)
            else:
                self.board.scatter(X=self.datapoints,
                                   Y=self.labels,
                                   opts=dict(title=win_name,
                                             markersize=5,
                                             ),
                                   win=win_name)

    def Save(self, path, name):
        pass


class Histogram(Post):
    def __init__(self, id, x, numbins=20, axis_x=None, axis_y=None, board=None):
        super().__init__(types={"charts"})
        self.id = id
        self.x = x
        self.numbins = numbins
        self.board = board
        if not axis_x:
            self.axis_x = "X"
        else:
            self.axis_x = axis_x

        if not axis_y:
            self.axis_y = "Y"
        else:
            self.axis_y = axis_y

    def post(self):
        if self.board is None or self.x.size <= 1:
            return

        self.board.histogram(X=self.x,
                             opts={'title': self.id,
                                   'numbins': self.numbins,
                                   'xlabel': self.axis_x,
                                   'ylabel': self.axis_y},
                             win=self.id)

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + "/" + name + '.csv', 'w') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow([self.axis_x] + self.id)
            bin_location = self.x.min()
            hist = np.histogram(self.x, bins=self.numbins)

            for csv_line in range(len(hist)):
                line_writer.writerow([bin_location] + hist[csv_line])


class Plot(Post):
    def __init__(self, id, labels, y, x=None, axis_x=None, axis_y=None, board=None):
        super().__init__(types={"charts"})
        self.id = id
        self.board = board
        if hasattr(y, '__iter__'):
            max_len = len(max(y, key=len))
            l = []
            for y_i in y:
                pad_length = max_len - len(y_i)
                l.append(np.pad(y_i, (0, pad_length), 'constant', constant_values=np.nan))
            self.y = np.vstack(l).transpose()
        else:
            self.y = y
            max_len = len(y)

        if x is None:
            self.x = np.arange(0, max_len)
        else:
            self.x = x

        self.labels = labels

        if not axis_x:
            self.axis_x = "X"
        else:
            self.axis_x = axis_x

        if not axis_y:
            self.axis_y = "Y"
        else:
            self.axis_y = axis_y

    def post(self):
        if self.board is None:
            return

        self.board.line(Y=self.y,
                        X=self.x,
                        opts={'title': self.id,
                              'legend': self.labels,
                              'xlabel': self.axis_x,
                              'ylabel': self.axis_y},
                        win=self.id)

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + "/" + name + '.csv', 'w') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow([self.axis_x] + self.labels)
            for csv_line in range(len(self.x)):
                line_writer.writerow(np.append([self.x[csv_line]], self.y[csv_line, :]))


class Graph(Post):
    def __init__(self, id, labels, axis_x=None, axis_y=None, window=-1, board=None):
        super().__init__(types={"charts"})
        self.id = id
        self.board = board
        self.x = None
        self.y = None
        self.window = window

        self.x_batch = np.array([])
        self.y_batch = np.array([])

        if not axis_x:
            self.axis_x = "X"
        else:
            self.axis_x = axis_x

        if not axis_y:
            self.axis_y = "Y"
        else:
            self.axis_y = axis_y

        self.labels = labels

    def Clear(self):
        self.x = None
        self.y = None

    def post(self):
        if self.board is None or self.x_batch.size <= 2:
            return

        if self.x is None:
            self.y = self.y_batch
            self.x = self.x_batch

            if self.window > 0 and self.x.shape[0] >= self.window:
                self.y = self.y[-self.window:, :]
                self.x = self.x[-self.window:]

            self.board.line(Y=self.y.squeeze(),
                            X=self.x,
                            opts={'title': self.id,
                                  'legend': self.labels,
                                  'xlabel': self.axis_x,
                                  'ylabel': self.axis_y},
                            win=self.id)
        else:
            self.y = np.vstack([self.y, self.y_batch])
            self.x = np.append(self.x, self.x_batch)

            if self.window > 0 and self.x.shape[0] >= self.window:
                self.y = self.y[-self.window:, :]
                self.x = self.x[-self.window:]
                self.board.line(Y=self.y.squeeze(),
                                X=self.x,
                                opts={'title': self.id,
                                      'legend': self.labels,
                                      'xlabel': self.axis_x,
                                      'ylabel': self.axis_y},
                                win=self.id)
            else:
                self.board.line(Y=self.y_batch.squeeze(),
                                X=self.x_batch,
                                win=self.id,
                                update='append')

        self.x_batch = np.array([])
        self.y_batch = np.array([])

    def Save(self, path, name):
        if self.x is None:
            return

        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + "/" + name + '.csv', 'w') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow([self.axis_x] + self.labels)
            for csv_line in range(len(self.x)):
                line_writer.writerow(np.append([self.x[csv_line]], self.y[csv_line, :]))

    def AddPoint(self, x, y):
        if self.x_batch.size == 0:
            self.y_batch = np.hstack(y)
            self.x_batch = np.array([x])
            return

        self.y_batch = np.vstack([self.y_batch, y])
        self.x_batch = np.append(self.x_batch, x)


class Images(Post):
    def __init__(self, id, imgs, scale=1.0, group=1, board=None):
        super().__init__(types={"multimedia"})
        self.id = id
        self.board = board
        self.group = len(imgs) // group
        self.imgs = []
        for img in imgs:
            img = np.squeeze(255 * img).astype(np.uint8)
            if scale is None:
                self.imgs.append(img)
            else:
                if img.ndim == 2:
                    self.imgs.append(cv2.resize(img, (int(scale * img.shape[-1]), int(scale * np.squeeze(img).shape[-2]))))
                else:
                    self.imgs.append(np.rollaxis(cv2.resize(np.rollaxis(img, 0, 3), (int(scale * img.shape[-1]),
                                                                                     int(scale * np.squeeze(img).shape[-2]))), 2))

    def post(self):
        if self.board is None:
            return
        if self.group > 0:
            self.board.images(self.imgs, opts=dict(title=self.id), win=self.id, nrow=self.group)
        else:
            self.board.images(self.imgs, opts=dict(title=self.id), win=self.id)

    def Save(self, path, name):
        folder_path = path + '/' + name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for idx, img in enumerate(self.imgs):
            if img.ndim == 2:
                cv2.imwrite(folder_path + '/' + str(idx) + '.jpg', img)
            else:
                cv2.imwrite(folder_path + '/' + str(idx) + '.jpg', cv2.cvtColor(np.rollaxis(img, 0, 3), cv2.COLOR_RGB2BGR))


class ImageAttentionMap(Post):
    def __init__(self, id, img, attention, focus=(0, 0), scale=1.0, board=None, alpha=0.5, increase_contrast=False):
        self.id = id
        super().__init__(types={"multimedia"})
        self.board = board
        self.img = np.squeeze(255 * img).astype(np.uint8)
        self.alpha = alpha

        attn_feature_scale = int(math.sqrt((img.shape[-2] * img.shape[-1]) // attention.shape[-1]))
        self.scale = scale
        self.feature_height = img.shape[-2] // attn_feature_scale
        self.feature_width = img.shape[-1] // attn_feature_scale

        if img.ndim == 2:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        else:
            self.img = np.rollaxis(self.img, 0, 3)  # Convert to opencv format H x W x C

        if scale != 1:
            self.img = cv2.resize(self.img, (int(scale * img.shape[-3]), int(scale * img.shape[-2])))  # In this case we store in opencv format H x W x C


        if increase_contrast:
            row_sums = attention.max(axis=1)
            att = attention / row_sums[:, np.newaxis]
        else:
            att = attention

        self.attention_map = att.reshape(self.feature_height, self.feature_width, self.feature_height, self.feature_width)

        self.pixel_focus = [focus[0], focus[1]]

    def _draw_attention_(self):
        heatmap_row = int(self.pixel_focus[0] * (self.feature_height - 1))
        heatmap_col = int(self.pixel_focus[1] * (self.feature_width - 1))

        pixel_coord = (int(self.pixel_focus[1] * (self.img.shape[1] - 1)), int(self.pixel_focus[0] * (self.img.shape[0] - 1)))

        heatmap = cv2.applyColorMap(cv2.resize((255 * self.attention_map[heatmap_row, heatmap_col]).astype(np.uint8),
                                               (int(self.img.shape[-3]), int(self.img.shape[-2]))), cv2.COLORMAP_TURBO)


        overlayed = heatmap * self.alpha + self.img * (1-self.alpha)
        overlayed = cv2.circle(overlayed, pixel_coord, 2, (0, 0, 0), thickness=1)
        return np.rollaxis(cv2.cvtColor(overlayed.astype(np.uint8), cv2.COLOR_BGR2RGB), 2) # Now we return an image in our image format CxHxW

    def update(self, event):
        if event['event_type'] == 'Click':
            self.pixel_focus[0] = max(min(event['image_coord']['y'], event['pane_data']['width']), 0) / event['pane_data']['width']
            self.pixel_focus[1] = max(min(event['image_coord']['x'], event['pane_data']['height']), 0) / event['pane_data']['height']

        overlayed_img = self._draw_attention_()
        self.board.image(overlayed_img, opts=dict(title=self.id), win=self.id)

    def post(self):
        if self.board is None or self.img.size == 0:
            return

        overlayed_img = self._draw_attention_()
        win = self.board.image(overlayed_img, opts=dict(title=self.id), win=self.id)

        if self.win is None:  # If we have not registered the callback do it now
            self.board.register_event_handler(self.update, win)
            self.win = win

    def Save(self, path, name):
        pass


class Image(Post):
    def __init__(self, id, img, scale=1.0, board=None):
        super().__init__(types={"multimedia"})
        self.id = id
        self.board = board
        img = np.squeeze(255 * img).astype(np.uint8)
        if scale is None:
            self.img = img
        else:
            if img.ndim == 2:
                self.img = cv2.resize(img, (int(scale * img.shape[-1]), int(scale * np.squeeze(img).shape[-2])))
            else:
                self.img = np.rollaxis(cv2.resize(np.rollaxis(img, 0, 3), (int(scale * img.shape[-1]), int(scale * np.squeeze(img).shape[-2]))), 2)

    def post(self):
        if self.board is None or self.img.size == 0:
            return
        self.board.image(self.img, opts=dict(title=self.id), win=self.id)

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.img.ndim == 2:
            cv2.imwrite(path + '/' + name + '.jpg', self.img)
        else:
            cv2.imwrite(path + '/' + name + '.jpg', cv2.cvtColor(np.rollaxis(self.img, 0, 3), cv2.COLOR_RGB2BGR))


class Table(Post):
    def __init__(self, id, headers, table_data=[], board=None):
        super().__init__(types={"parameters"})
        self.id = id
        self.board = board
        self.headers = headers
        self.table = table_data

    def Load(self, table_data):
        self.table = table_data

    def Clear(self):
        self.table.clear()

    def AddRow(self, row):
        if not self.table:
            self.table = [row]
        else:
            self.table.append(row)

    def post(self):
        if self.board is None:
            return

        htmlcode = table(self.table, header_row=self.headers, style="width:100%")
        self.board.text(htmlcode, win=self.id)

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + "/" + name + '.csv', 'w') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow(self.headers)
            for csv_line in self.table:
                line_writer.writerow(csv_line)
        pass


class Audio(Post):
    def __init__(self, id, audio=np.array([]), rate=50000, spectrogram=False, board=None):
        super().__init__(types={"multimedia"})
        self.id = id
        self.board = board
        self.audio = ((2 ** 15) * audio).astype(np.int16)
        self.rate = rate
        self.spectrogram = spectrogram
        if self.spectrogram:
            self.freq, self.sample_time, self.Sxx = signal.spectrogram(self.audio, self.rate)

    def post(self):
        if self.board is None:
            return
        temp_file = filify(self.board.env) + "_" + filify(self.id)
        self.Save("/tmp", temp_file)
        full_path = "/tmp/" + temp_file + '.wav'
        opts = dict(sample_frequency=self.rate)
        self.board.audio(audiofile=full_path, win=self.id, opts=opts)
        if self.spectrogram:
            self.post_spectrogram()

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        wav.write(path + '/' + name + ".wav", self.rate, self.audio)

    def post_spectrogram(self):
        if self.board is None:
            return

        spectrogram_id = "Spectrogram of " + self.id
        time_freq_map = {"rownames": self.freq.tolist(), "columnnames": self.sample_time.tolist(), "title": spectrogram_id}
        self.board.heatmap(X=self.Sxx, opts=time_freq_map, win=spectrogram_id)


class Video(Post):
    def __init__(self, id, video=np.array([]), fps=25, audio=None, rate=50000, ffmpeg_experimental=False, board=None):
        super().__init__(types={"multimedia"})
        self.id = id
        self.board = board
        if video.size == 0:
            self.video = []
        else:
            self.video = []
            self.Load(video)

        self.fps = fps
        self.audio = audio
        self.rate = rate
        self.ffmpeg_experimental = ffmpeg_experimental

    def Clear(self):
        self.video = []

    def Load(self, video):
        video[video > 1.0] = 1.0
        video[video < 0.0] = 0.0
        for frame in range(video.shape[0]):
            self.video.append(video[frame, :, :, :])

    def AddFrame(self, frame):
        self.video.append(frame)

    def post(self):
        if self.board is None or len(self.video) < 1:
            return

        temp_file = filify(self.board.env) + "_" + filify(self.id)
        if not self.Save("/tmp", temp_file):
            return

        full_path = "/tmp/" + temp_file + '.mp4'

        opts = dict(fps=self.fps)
        self.board.video(videofile=full_path, win=self.id, opts=opts)

    def Save(self, path, name, gif=False, extension=".mp4"):
        success = True
        if not os.path.exists(path):
            os.makedirs(path)

        video_path = path + '/' + name
        if gif:
            video_path += '.gif'
            gif_frames = []
            for single_frame in self.video:
                gif_frames.append(np.rollaxis(single_frame, 0, 3))
            imageio.mimsave(path + '/' + name + '.gif', gif_frames, fps=self.fps)
        else:
            video_path += extension
            if self.audio is None:
                menpo.io.export_video([menpo.image.Image(frame, copy=False) for frame in self.video],
                                      video_path, fps=self.fps, overwrite=True)
            else:
                temp_filename = next(tempfile._get_candidate_names())
                menpo.io.export_video([menpo.image.Image(frame, copy=False) for frame in self.video],
                                      "/tmp/" + temp_filename + ".mp4", fps=self.fps, overwrite=True)
                wav.write("/tmp/" + temp_filename + ".wav", self.rate, self.audio)

                try:
                    in1 = ffmpeg.input("/tmp/" + temp_filename + ".mp4")
                    in2 = ffmpeg.input("/tmp/" + temp_filename + ".wav")

                    if self.ffmpeg_experimental:
                        out = ffmpeg.output(in1['v'], in2['a'], video_path, strict='-2',
                                            loglevel="panic").overwrite_output()
                    else:
                        out = ffmpeg.output(in1['v'], in2['a'], video_path, loglevel="panic").overwrite_output()
                    out.run(quiet=True)
                except:
                    success = False

                if os.path.isfile("/tmp/" + temp_filename + ".mp4"):
                    os.remove("/tmp/" + temp_filename + ".mp4")
                if os.path.isfile("/tmp/" + temp_filename + ".wav"):
                    os.remove("/tmp/" + temp_filename + ".wav")

        return success


class JointAnimation(Post):
    def __init__(self, id, points=np.array([]), edges=[], fps=25, audio=None, rate=50000, order=None, colour=None, ffmpeg_experimental=False,
                 board=None):
        super().__init__(types={"multimedia"})
        self.id = id
        self.board = board
        self.points = points.copy()
        if edges == "face":
            self.edges = FACE_EDGES
        else:
            self.edges = edges

        if colour is None:
            self.colour = (255, 0, 0)
        else:
            self.colour = colour

        self.fps = int(fps)
        self.audio = audio
        self.rate = rate
        self.max_canvas = []
        self.min_canvas = []

        if order is not None:
            for i in range(len(order)):
                self.points[:, :, i] = points[:, :, order[i]]

        self._perform_checks_()
        self.ffmpeg_experimental = ffmpeg_experimental

    def clear(self):
        self.points = np.array([])

    def _perform_checks_(self):
        if self.points.size != 0:

            for i in range(2):
                self.max_canvas.append(np.amax(self.points[:, :, i]))
                self.min_canvas.append(np.amin(self.points[:, :, i]))

        if self.points.ndim == 3 and self.points.shape[2] > 3:
            warnings.warn("points have dimension larger than 3", RuntimeWarning)

    def add_frame(self, frame):
        self.points = np.vstack([self.points, frame])
        self._perform_checks_()

    def add_audio(self, audio=None, rate=50000):
        self.audio = audio
        self.rate = rate

    def load(self, landmarks, dim=2, order=None):
        with open(landmarks, 'rt', encoding="ascii") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')

            seq_landmarks = None
            for frame_no, landmarks in enumerate(csvreader):
                landmark = np.zeros([1, len(landmarks) // dim, dim])
                for point in range(1, len(landmarks), dim):
                    for i, j in enumerate(order):
                        landmark[0, point // dim, i] = int(landmarks[point + j])

                if seq_landmarks is None:
                    seq_landmarks = landmark
                else:
                    seq_landmarks = np.vstack([seq_landmarks, landmark])

        self.points = seq_landmarks
        self._perform_checks_()

    def post(self):
        if self.board is None:
            return

        temp_file = filify(self.board.env) + "_" + filify(self.id)
        if not self.Save("/tmp", temp_file):
            return

        full_path = "/tmp/" + temp_file + '.mp4'

        opts = dict(fps=self.fps)
        self.board.video(videofile=full_path, win=self.id, opts=opts)

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        width = int(self.max_canvas[0] - self.min_canvas[0])
        height = int(self.max_canvas[1] - self.min_canvas[1])

        if self.audio is None:
            filename = path + "/" + name + ".mp4"
        else:
            filename = "/tmp/" + next(tempfile._get_candidate_names()) + ".mp4"

        video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), float(self.fps),
                                (height, width))
        for frame in self.points:
            frame = frame - np.array([self.min_canvas[0], self.min_canvas[1]])
            canvas = np.ones([width, height, 3])
            canvas *= (255, 255, 255)  # canvas is by default white

            for node in frame:
                cv2.circle(canvas, (int(node[0]), int(node[1])), 2, self.colour, -1)

            for edge in self.edges:
                cv2.line(canvas,
                         (int(frame[edge[0]][0]), int(frame[edge[0]][1])),
                         (int(frame[edge[1]][0]), int(frame[edge[1]][1])),
                         self.colour, 1)

            video.write(canvas.astype('uint8'))
        video.release()

        if self.audio is not None:
            wav.write(swp_extension(filename, ".wav"), self.rate, self.audio)
            try:
                in1 = ffmpeg.input(filename)
                in2 = ffmpeg.input(swp_extension(filename, ".wav"))

                if self.ffmpeg_experimental:
                    out = ffmpeg.output(in1['v'], in2['a'], path + "/" + name + ".mp4", strict='-2',
                                        loglevel="panic").overwrite_output()
                else:
                    out = ffmpeg.output(in1['v'], in2['a'], path + "/" + name + ".mp4",
                                        loglevel="panic").overwrite_output()
                out.run(quiet=True)
            except:
                warnings.warn("Problem mixing video and audio", RuntimeWarning)

            if os.path.isfile(filename):
                os.remove(filename)
            if os.path.isfile(swp_extension(filename, ".wav")):
                os.remove(swp_extension(filename, ".wav"))


class AdjustableParameter():
    def __init__(self, init):
        self.value = float(init)

    def as_float(self):
        return self.value

    def as_int(self):
        return int(self.value)

    def update(self, value):
        self.value = float(value)

    def scale(self, scale):
        self.value = scale * self.value


class Bulletin():
    def __init__(self, server='http://localhost', save_path='.', env='main', ffmpeg_experimental=False, username=None, password=None,
                 interactive=False):
        self.vis = visdom.Visdom(env=env, server=server, username=username, password=password)
        self.Posts = {}
        self.save_path = save_path
        self.ffmpeg_experimental = ffmpeg_experimental
        self.interactive = interactive
        self.controlled_variables = [self.clear_message]
        if self.interactive:
            self.properties = [{'type': 'button', 'name': 'Message', 'value': 'Hello!'}]
            self.callbacks = [self.clear_message]
            self.properties_window = self.vis.properties(self.properties)
            self.vis.register_event_handler(self._control_window_callback_, self.properties_window)

    def clear_message(self, event):
        return "Messages Cleared!"

    def _control_window_callback_(self, event):
        if event['event_type'] == 'PropertyUpdate':
            prop_id = event['propertyId']
            value = event['value']
            message = self.callbacks[prop_id](value)
            if message is not None:
                self.properties[0]['value'] = message
            elif prop_id:
                self.properties[prop_id]['value'] = value

            self.vis.properties(self.properties, win=self.properties_window)

    def DeleteItem(self, id):
        self.Posts.pop(id)

    def RemoveItemFromBulletin(self, id):
        del self.Posts[id]

    def ClearBulletin(self):
        self.Posts.clear()

    def add_text_control(self, label, callback, initial=""):
        self.properties += [{'type': 'text', 'name': label, 'value': initial}]
        self.callbacks += [callback]
        self.vis.properties(self.properties, win=self.properties_window)

    def add_adjustable_parameter(self, label, initial=0):
        self.controlled_variables.append(AdjustableParameter(initial))
        if self.interactive:
            self.properties += [{'type': 'number', 'name': label, 'value': str(initial)}]
            self.callbacks += [self.controlled_variables[-1].update]
            self.vis.properties(self.properties, win=self.properties_window)
        return self.controlled_variables[-1]

    def create_joint_animation(self, id, points, edges=None, fps=25, audio=None, rate=50000, order=None, colour=None):
        self.Posts[id] = JointAnimation(id, points, edges, fps, audio, rate, order, colour, ffmpeg_experimental=self.ffmpeg_experimental,
                                        board=self.vis)
        return self.Posts[id]

    def create_image_attention(self, id, image, attention, scale=1.0, focus=(0, 0), alpha=0.5, increase_contrast=False):
        if id in self.Posts.keys():
            self.vis.clear_event_handlers(self.Posts[id].win)

        self.Posts[id] = ImageAttentionMap(id, image, attention, focus=focus, scale=scale, board=self.vis, alpha=alpha, increase_contrast=increase_contrast)
        return self.Posts[id]

    def CreateImage(self, id, image, scale=1.0):
        self.Posts[id] = Image(id, image, scale=scale, board=self.vis)
        return self.Posts[id]

    def CreateImageList(self, id, images, scale=1.0, group=1):
        self.Posts[id] = Images(id, images, scale=scale, group=group, board=self.vis)
        return self.Posts[id]

    def CreateAudio(self, id, audio, rate=50000, spectrogram=False):
        self.Posts[id] = Audio(id, audio, rate, spectrogram=spectrogram, board=self.vis)
        return self.Posts[id]

    def CreateVideo(self, id, video=np.array([]), fps=25, audio=None, rate=50000):
        self.Posts[id] = Video(id, video, fps, audio, rate, ffmpeg_experimental=self.ffmpeg_experimental, board=self.vis)
        return self.Posts[id]

    def CreateTable(self, id, headers, table_data=[]):
        self.Posts[id] = Table(id, headers, table_data, board=self.vis)
        return self.Posts[id]

    def CreatePlot(self, id, labels, y, x=None, axis_x=None, axis_y=None):
        self.Posts[id] = Plot(id, labels, y, x, axis_x, axis_y, board=self.vis)
        return

    def CreateGraph(self, id, labels, axis_x=None, axis_y=None, window=-1):
        self.Posts[id] = Graph(id, labels, axis_x, axis_y, window, board=self.vis)
        return self.Posts[id]

    def CreateHistogram(self, id, x, numbins=20, axis_x=None, axis_y=None):
        self.Posts[id] = Histogram(id, x, numbins, axis_x, axis_y, board=self.vis)
        return self.Posts[id]

    def CreateScatterPlot(self, id, datapoints, labels=None, sequence_coloring=True, t_sne=False, perplexity=30, iterations=10000):
        self.Posts[id] = Scatter(id, datapoints, labels, sequence_coloring, t_sne, perplexity, iterations, board=self.vis)
        return self.Posts[id]

    def Post(self):
        for post_id in self.Posts:
            if post_id != None:
                try:
                    self.Posts[post_id].post()
                except Exception as e:
                    warnings.warn("Couldn't post to bulletin: " + str(e), RuntimeWarning)
            else:
                del self.Posts[post]

    def SaveState(self, save_path=None, filter=None):
        if save_path is None:
            save_path = self.save_path
        for post in self.Posts:
            if post != None:
                if filter is None or bool(self.Posts[post].types.intersection(filter)):
                    self.Posts[post].Save(save_path, filify(post))
            else:
                del self.Posts[post]
