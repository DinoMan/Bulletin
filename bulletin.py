import scipy.misc
import visdom
import numpy as np
from sklearn.manifold import TSNE
import csv
import matplotlib.cm as cm
import imageio
import menpo
import os


class Scatter:
    def __init__(self, datapoints, labels=None, sequence_coloring=True, t_sne=False, perplexity=30, iterations=10000):
        self.datapoints = datapoints
        self.no_points = datapoints.shape[0]
        self.labels = labels
        self.sequence_coloring = sequence_coloring
        if t_sne:
            self.TSNE = TSNE(n_components=2, perplexity=perplexity, n_iter=iterations)
        else:
            self.TSNE = None

    def _Post(self, board, id):
        if self.TSNE is None:
            mapped_datapoints = self.datapoints
        else:
            if self.datapoints.size <= 2:
                return
            mapped_datapoints = self.TSNE.fit_transform(self.datapoints)

        options = {'title': id,
                   'markersize': 10}

        if self.labels is None:
            if self.sequence_coloring:
                colors = 255 * (cm.coolwarm(np.arange(0, 1, step=1.0 / self.no_points))[:, :3])
                options.update({'markercolor': colors.astype(int)})

            board.scatter(X=mapped_datapoints,
                          opts=options, win=id)
        else:
            board.scatter(X=mapped_datapoints,
                          Y=self.labels,
                          opts=options,
                          win=id)


class Histogram:
    def __init__(self, x, numbins=20, axis_x=None, axis_y=None):
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
        bin_size = (self.x.max() - self.x.min()) / self.numbins
        with open(path + "/" + name + '.csv', 'wb') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow([self.axis_x] + id)
            bin_location = self.x.min()
            hist = np.histogram(self.x, bins=self.numbins)

            for csv_line in range(len(hist)):
                line_writer.writerow([bin_location] + hist[csv_line])


class Graph:
    def __init__(self, labels, axis_x=None, axis_y=None, window=-1):
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

    def _Post(self, board, id):
        if self.x_batch.size <= 2:
            return

        if self.x is None:
            self.y = self.y_batch
            self.x = self.x_batch

            if self.window > 0 and self.x.shape[0] >= self.window:
                self.y = self.y[-self.window:, :]
                self.x = self.x[-self.window:]

            board.line(Y=self.y,
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
                board.line(Y=self.y,
                           X=self.x,
                           opts={'title': id,
                                 'legend': self.labels,
                                 'xlabel': self.axis_x,
                                 'ylabel': self.axis_y},
                           win=id)
            else:
                # Lines added to make use of the fast update, which however has a bug. It is fixed in the recent trunk
                if self.y_batch.ndim == 2 and self.x_batch.ndim == 1:
                    X = np.tile(self.x_batch, (self.y_batch.shape[1], 1)).transpose()

                board.line(Y=self.y_batch,
                           X=X,
                           win=id,
                           update='append')

        self.x_batch = np.array([])
        self.y_batch = np.array([])

    def Save(self, path, name):
        with open(path + "/" + name + '.csv', 'wb') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
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


class Image:
    def __init__(self, img, scale=2.0):
        if scale is None:
            self.img = img
        else:
            self.img = np.rollaxis(scipy.misc.imresize(img, scale), 2, 0)

    def _Post(self, board, id):
        if self.img.size == 0:
            return

        board.image(self.img, opts=dict(title=id), win=id)

    def Save(self, path, name):
        scipy.misc.imsave(path + '/' + name + '.jpg', np.rollaxis(self.img, 0, 3))


class Video:
    def __init__(self, video=np.array([])):
        if video.size == 0:
            self.video = []
        else:
            self.Load(video)

    def Clear(self):
        self.video = []

    def Load(self, video):
        video[video > 1] = 1.0
        video[video < 0] = 0.0
        for frame in range(video.shape[0]):
            self.video.append(video[frame, :, :, :])

    def AddFrame(self, frame):
        self.video.append(frame)

    def _Post(self, board, id):
        if len(self.video) < 1:
            return

            # TODO Video doesn't work ATM (Could just be the lab computer has issues with ffmpeg though)
            # board.video(np.stack(self.video), win=id)

    def Save(self, path, name, fps=15, gif=False):
        if not os.path.exists(path):
            os.makedirs(path)

        if gif:
            gif_frames = []
            for single_frame in self.video:
                gif_frames.append(np.rollaxis(single_frame, 0, 3))
            imageio.mimsave(path + '/' + name + '.gif', gif_frames, fps=fps)
        else:
            menpo.image.Image
            menpo.io.export_video([menpo.image.Image(frame, copy=False) for frame in self.video],
                                  path + '/' + name + '.mp4', fps=fps, overwrite=True)


class Bulletin():
    def __init__(self, save_path='.', env='main'):
        self.vis = visdom.Visdom(env=env)
        self.Posts = {}
        self.save_path = save_path

    def DeleteItem(self, id):
        self.Posts.pop(id)

    def RemoveItemFromBulletin(self, id):
        del self.Posts[id]

    def ClearBulletin(self):
        self.Posts.clear()

    def CreateImage(self, image, id):
        self.Posts[id] = Image(image)
        return self.Posts[id]

    def CreateVideo(self, id, video=np.array([])):
        self.Posts[id] = Video(video)
        return self.Posts[id]

    def CreateGraph(self, id, labels, axis_x=None, axis_y=None, window=-1):
        self.Posts[id] = Graph(labels, axis_x, axis_y, window)
        return self.Posts[id]

    def CreateHistogram(self, id, x, numbins=20, axis_x=None, axis_y=None):
        self.Posts[id] = Histogram(x, numbins, axis_x, axis_y)
        return self.Posts[id]

    def CreateScatterPlot(self, id, datapoints, labels=None, sequence_coloring=True, t_sne=False, perplexity=30,
                          iterations=10000):
        self.Posts[id] = Scatter(datapoints, labels, sequence_coloring, t_sne, perplexity, iterations)
        return self.Posts[id]

    def Post(self):
        for post in self.Posts:
            if post != None:
                self.Posts[post]._Post(self.vis, post)
            else:
                del self.Posts[post]

    def SaveState(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        for post in self.Posts:
            if post != None:
                self.Posts[post].Save(save_path, post.replace(" ", "_"))
            else:
                del self.Posts[post]
