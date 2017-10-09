import scipy.misc
import visdom
import numpy as np
import os
import csv

class Graph:
    def __init__(self, labels, axis_x = None, axis_y = None):
        self.x = np.array([])
        if not axis_x:
            self.axis_x = "X"
        else:
            self.axis_x = axis_x

        if not axis_x:
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
                   win=2)

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
