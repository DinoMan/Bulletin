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
    def __init__(self, datapoints, labels=None, sequence_coloring=True, t_sne=False, perplexity=10,
                 iterations=2000, filter_name=None):
        super().__init__(types={"charts"})
        self.no_points = datapoints.shape[0]
        self.label_mapping = {}
        self.labels = labels
        self.names = None
        self.filter_name = filter_name

        self.ChangeLabelling(labels)

        self.sequence_coloring = sequence_coloring
        if t_sne:
            TSNE_Mapper = TSNE(n_components=2, perplexity=perplexity, n_iter=iterations)
            self.datapoints = TSNE_Mapper.fit_transform(datapoints)
        else:
            self.datapoints = datapoints

    def ChangeLabelling(self, labels, filter_name=None):
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

    def _Post(self, board, id):
        win_name = id
        if self.filter_name is not None:
            win_name += "Filtered by: " + self.filter_name

        if self.labels is None:
            if self.sequence_coloring:
                colors = 255 * (cm.coolwarm(np.arange(0, 1, step=1.0 / self.no_points))[:, :3])
            board.scatter(X=self.datapoints,
                          opts=dict(title=id,
                                    markercolor=colors.astype(int),
                                    markersize=5,
                                    ),
                          win=win_name)
        else:
            if self.names is not None:
                board.scatter(X=self.datapoints,
                              Y=self.labels,
                              opts=dict(title=win_name,
                                        legend=self.names,
                                        markersize=5,
                                        ),
                              win=win_name)
            else:
                board.scatter(X=self.datapoints,
                              Y=self.labels,
                              opts=dict(title=win_name,
                                        markersize=5,
                                        ),
                              win=win_name)

    def Save(self, path, name):
        pass


class Histogram(Post):
    def __init__(self, x, numbins=20, axis_x=None, axis_y=None):
        super().__init__(types={"charts"})
        self.x = x
        self.numbins = numbins

        if not axis_x:
            self.axis_x = "X"
        else:
            self.axis_x = axis_x

        if not axis_y:
            self.axis_y = "Y"
        else:
            self.axis_y = axis_y

    def _Post(self, board, id):
        if self.x.size <= 1:
            return

        board.histogram(X=self.x,
                        opts={'title': id,
                              'numbins': self.numbins,
                              'xlabel': self.axis_x,
                              'ylabel': self.axis_y},
                        win=id)

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + "/" + name + '.csv', 'w') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow([self.axis_x] + id)
            bin_location = self.x.min()
            hist = np.histogram(self.x, bins=self.numbins)

            for csv_line in range(len(hist)):
                line_writer.writerow([bin_location] + hist[csv_line])


class Plot(Post):
    def __init__(self, labels, y, x=None, axis_x=None, axis_y=None):
        super().__init__(types={"charts"})
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

    def _Post(self, board, id):

        board.line(Y=self.y,
                   X=self.x,
                   opts={'title': id,
                         'legend': self.labels,
                         'xlabel': self.axis_x,
                         'ylabel': self.axis_y},
                   win=id)

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + "/" + name + '.csv', 'w') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow([self.axis_x] + self.labels)
            for csv_line in range(len(self.x)):
                line_writer.writerow(np.append([self.x[csv_line]], self.y[csv_line, :]))


class Graph(Post):
    def __init__(self, labels, axis_x=None, axis_y=None, window=-1):
        super().__init__(types={"charts"})
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

    def _Post(self, board, id):
        if self.x_batch.size <= 2:
            return

        if self.x is None:
            self.y = self.y_batch
            self.x = self.x_batch

            if self.window > 0 and self.x.shape[0] >= self.window:
                self.y = self.y[-self.window:, :]
                self.x = self.x[-self.window:]

            board.line(Y=self.y.squeeze(),
                       X=self.x,
                       opts={'title': id,
                             'legend': self.labels,
                             'xlabel': self.axis_x,
                             'ylabel': self.axis_y},
                       win=id)
        else:
            self.y = np.vstack([self.y, self.y_batch])
            self.x = np.append(self.x, self.x_batch)

            if self.window > 0 and self.x.shape[0] >= self.window:
                self.y = self.y[-self.window:, :]
                self.x = self.x[-self.window:]
                board.line(Y=self.y.squeeze(),
                           X=self.x,
                           opts={'title': id,
                                 'legend': self.labels,
                                 'xlabel': self.axis_x,
                                 'ylabel': self.axis_y},
                           win=id)
            else:
                board.line(Y=self.y_batch.squeeze(),
                           X=self.x_batch,
                           win=id,
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
    def __init__(self, imgs, scale=1.0, group=1):
        super().__init__(types={"multimedia"})
        self.group = len(imgs) // group
        self.imgs = []
        for img in imgs:
            img = np.squeeze(255 * img).astype(np.uint8)
            if scale is None:
                self.imgs.append(img)
            else:
                if img.ndim == 2:
                    self.imgs.append(cv2.resize(img, (int(scale * img.shape[0]), int(scale * np.squeeze(img).shape[1]))))
                else:
                    self.imgs.append(np.swapaxes(cv2.resize(np.swapaxes(img, 0, 2), (int(scale * img.shape[1]),
                                                                                     int(scale * np.squeeze(img).shape[2]))), 0, 2))

    def _Post(self, board, id):
        if self.group > 0:
            board.images(self.imgs, opts=dict(title=id), win=id, nrow=self.group)
        else:
            board.images(self.imgs, opts=dict(title=id), win=id)

    def Save(self, path, name):
        folder_path = path + '/' + name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for idx, img in enumerate(self.imgs):
            if img.ndim == 2:
                cv2.imwrite(folder_path + '/' + str(idx) + '.jpg', img)
            else:
                cv2.imwrite(folder_path + '/' + str(idx) + '.jpg', cv2.cvtColor(np.rollaxis(img, 0, 3), cv2.COLOR_RGB2BGR))


class Image(Post):
    def __init__(self, img, scale=1.0):
        super().__init__(types={"multimedia"})
        img = np.squeeze(255 * img).astype(np.uint8)
        if scale is None:
            self.img = img
        else:
            if img.ndim == 2:
                self.img = cv2.resize(img, (int(scale * img.shape[0]), int(scale * np.squeeze(img).shape[1])))
            else:
                self.img = np.swapaxes(cv2.resize(np.swapaxes(img, 0, 2), (int(scale * img.shape[1]), int(scale * np.squeeze(img).shape[2]))), 0, 2)

    def _Post(self, board, id):
        if self.img.size == 0:
            return
        board.image(self.img, opts=dict(title=id), win=id)

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.img.ndim == 2:
            cv2.imwrite(path + '/' + name + '.jpg', self.img)
        else:
            cv2.imwrite(path + '/' + name + '.jpg', cv2.cvtColor(np.rollaxis(self.img, 0, 3), cv2.COLOR_RGB2BGR))


class Table(Post):
    def __init__(self, headers, table_data=[]):
        super().__init__(types={"parameters"})
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

    def _Post(self, board, id):
        htmlcode = table(self.table, header_row=self.headers, style="width:100%")
        board.text(htmlcode, win=id)

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
    def __init__(self, audio=np.array([]), rate=50000, spectrogram=False):
        super().__init__(types={"multimedia"})
        self.audio = ((2 ** 15) * audio).astype(np.int16)
        self.rate = rate
        self.spectrogram = spectrogram
        if self.spectrogram:
            self.freq, self.sample_time, self.Sxx = signal.spectrogram(self.audio, self.rate)

    def _Post(self, board, id):
        temp_file = filify(board.env) + "_" + filify(id)
        self.Save("/tmp", temp_file)
        full_path = "/tmp/" + temp_file + '.wav'
        opts = dict(sample_frequency=self.rate)
        board.audio(audiofile=full_path, win=id, opts=opts)
        if self.spectrogram:
            self._post_spectrogram_(board, id)

    def Save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        wav.write(path + '/' + name + ".wav", self.rate, self.audio)

    def _post_spectrogram_(self, board, id):
        spectrogram_id = "Spectrogram of " + id
        time_freq_map = {"rownames": self.freq.tolist(), "columnnames": self.sample_time.tolist(),
                         "title": spectrogram_id}
        board.heatmap(X=self.Sxx, opts=time_freq_map, win=spectrogram_id)


class Video(Post):
    def __init__(self, video=np.array([]), fps=25, audio=None, rate=50000, ffmpeg_experimental=False):
        super().__init__(types={"multimedia"})
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

    def _Post(self, board, id):
        if len(self.video) < 1:
            return

        temp_file = filify(board.env) + "_" + filify(id)
        if not self.Save("/tmp", temp_file):
            return

        full_path = "/tmp/" + temp_file + '.mp4'

        opts = dict(fps=self.fps)
        board.video(videofile=full_path, win=id, opts=opts)

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
    def __init__(self, points=np.array([]), edges=[], fps=25, audio=None, rate=50000,
                 order=None, colour=None, ffmpeg_experimental=False):
        super().__init__(types={"multimedia"})
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

    def _Post(self, board, id):
        temp_file = filify(board.env) + "_" + filify(id)
        if not self.Save("/tmp", temp_file):
            return

        full_path = "/tmp/" + temp_file + '.mp4'

        opts = dict(fps=self.fps)
        board.video(videofile=full_path, win=id, opts=opts)

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
        self.Posts[id] = JointAnimation(points, edges, fps, audio, rate, order, colour,
                                        ffmpeg_experimental=self.ffmpeg_experimental)
        return self.Posts[id]

    def CreateImage(self, id, image, scale=1.0):
        self.Posts[id] = Image(image, scale=scale)
        return self.Posts[id]

    def CreateImageList(self, id, images, scale=1.0, group=0):
        self.Posts[id] = Images(images, scale=scale, group=group)
        return self.Posts[id]

    def CreateAudio(self, id, audio, rate=50000, spectrogram=False):
        self.Posts[id] = Audio(audio, rate, spectrogram=spectrogram)
        return self.Posts[id]

    def CreateVideo(self, id, video=np.array([]), fps=25, audio=None, rate=50000):
        self.Posts[id] = Video(video, fps, audio, rate, ffmpeg_experimental=self.ffmpeg_experimental)
        return self.Posts[id]

    def CreateTable(self, id, headers, table_data=[]):
        self.Posts[id] = Table(headers, table_data)
        return self.Posts[id]

    def CreatePlot(self, id, labels, y, x=None, axis_x=None, axis_y=None):
        self.Posts[id] = Plot(labels, y, x, axis_x, axis_y)
        return

    def CreateGraph(self, id, labels, axis_x=None, axis_y=None, window=-1):
        self.Posts[id] = Graph(labels, axis_x, axis_y, window)
        return self.Posts[id]

    def CreateHistogram(self, id, x, numbins=20, axis_x=None, axis_y=None):
        self.Posts[id] = Histogram(x, numbins, axis_x, axis_y)
        return self.Posts[id]

    def CreateScatterPlot(self, id, datapoints, labels=None, sequence_coloring=True, t_sne=False,
                          perplexity=30, iterations=10000):
        self.Posts[id] = Scatter(datapoints, labels, sequence_coloring, t_sne, perplexity, iterations)
        return self.Posts[id]

    def Post(self):
        for post in self.Posts:
            if post != None:
                try:
                    self.Posts[post]._Post(self.vis, post)
                except:
                    warnings.warn("Couldn't post to bulletin", RuntimeWarning)
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
