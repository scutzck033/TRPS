from yellowbrick.features.manifold import Manifold
from yellowbrick.features.pca import PCADecomposition
import os
import numpy as np

class Visualization(object):
    def __init__(self,x,y,title):
        self.x = np.array(x)
        self.y = np.array(y)
        self.title = title

    def poof(self,visualizer,path,fileName,save):
        if save:
            if path and fileName:
                if not os.path.exists(path):
                    os.mkdir(path)
                visualizer.poof(outpath=path+fileName)
            else:
                print("please enter the base_path and file_name")
        else:
            visualizer.poof()

    def pca_visualization(self,path=None,fileName=None,save=False,
                          scale=True, center=False):
        visualizer = PCADecomposition(scale=scale,center=center,
                                      color=self.y,title=self.title)
        visualizer.fit_transform(self.x, self.y)
        self.poof(visualizer, path, fileName, save)

    def manifold_visualization(self,path=None,fileName=None,save=False,
                               manifold='mds', target='discrete'):
        visualizer = Manifold(manifold=manifold, target=target,title=self.title)
        visualizer.fit(self.x,self.y)
        self.poof(visualizer,path,fileName,save)


