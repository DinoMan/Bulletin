import scipy.misc
import visdom
import numpy as np
from sklearn.manifold import TSNE
import os
import csv
import matplotlib.cm as cm

class Scatter:
    def __init__(self, datapoints, labels = None, sequence_coloring = True, t_sne = False, perplexity=30, iterations=10000):
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
                colors = 255*(cm.coolwarm(np.arange(0, 1, step=1.0/self.no_points))[:,:3])
                options.update({'markercolor': colors.astype(int)})

            board.scatter(X=mapped_datapoints,
                          opts=options, win=id)
        else:
            board.scatter(X=mapped_datapoints,
                          Y=self.labels,
                          opts=options,
                          win=id)


class Histogram:
    def __init__(self, x, numbins = 20, axis_x = None, axis_y = None):
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
        bin_size = (x.max() - x.min())/self.numbins
        with open(path +"/" + name + '.csv', 'wb') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow( [self.axis_x] + id)
            bin_location = x.min()
            hist = np.histogram(a, bins=self.numbins)

            for csv_line in range(len(hist)):
                line_writer.writerow([bin_location] +  hist[csv_line])

class Graph:
    def __init__(self, labels, axis_x = None, axis_y = None):
        self.x = np.array([])
        if not axis_x:
            self.axis_x = "X"
        else:
            self.axis_x = axis_x

        if not axis_y:
            self.axis_y = "Y"
        else:
            self.axis_y = axis_y

        self.y = np.array([])
        self.labels = labels

    def _Post(self, board, id):
        if self.x.size <= 1:
            return

        board.line(Y=self.y,
                   X=self.x,
                   opts={'title': id,
                         'legend': self.labels,
                         'xlabel': self.axis_x,
                         'ylabel': self.axis_y},
                   win=id)

    def Save(self, path, name):
        with open(path +"/" + name + '.csv', 'wb') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow( [self.axis_x] + self.labels)
            for csv_line in range(len(self.x)):
                line_writer.writerow(np.append([self.x[csv_line]], self.y[csv_line, :]))

    def AddPoint(self, x, y):
        if self.x.size == 0:
            self.y = np.hstack(y)
            self.x = np.array([x])
            return

        self.y = np.vstack([self.y, y])
        self.x = np.append(self.x, x)

class Image:
    def __init__(self, img = np.array([])):
        self.img = img

    def _Post(self, board, id):
        if self.img.size == 0:
            return

        board.image(self.img, opts=dict(title=id), win=id)

    def Save(self, path, name):
        scipy.misc.imsave(path + "/" + name + '.jpg', self.img)

class Bulletin():
    def __init__(self, save_path = '.', env = 'main'):
        self.vis = visdom.Visdom(env=env)
        self.Posts = {}
        self.save_path = save_path

    def CreateImage(self, image, id):
        self.Posts[id] = Image(image)

    def CreateGraph(self, id, labels, axis_x = None, axis_y = None):
        self.Posts[id] = Graph(labels, axis_x, axis_y)
        return self.Posts[id]

    def CreateHistogram(self, id, x, numbins = 20, axis_x = None, axis_y = None):
        self.Posts[id] = Histogram(x, numbins, axis_x, axis_y)

    def CreateScatterPlot(self, id, datapoints, labels = None, sequence_coloring = True, t_sne = False, perplexity=30, iterations=10000):
        self.Posts[id] = Scatter(datapoints, labels, sequence_coloring, t_sne , perplexity, iterations)

    def Post(self):
        for post in self.Posts:
            if post != None:
                self.Posts[post]._Post(self.vis, post)
            else:
                del self.Posts[post]

    def SaveState(self):
        for post in self.Posts:
            if post != None:
                self.Posts[post].Save(self.save_path, post)
            else:
                del self.Posts[post]
