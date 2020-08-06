"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 3D array.
"""

import numpy as np
import matplotlib.pyplot as plt



def show_3d_images( input_matrix,upper_bound=0 ):

    if np.any(np.iscomplex(input_matrix)):
        mat = np.log( np.abs( input_matrix)+1)
    elif (upper_bound >0):
        mat=input_matrix
        mat[mat>upper_bound]=upper_bound
    else:
        mat = input_matrix
    

    if np.amin(mat) != 0:
        mat = mat - np.amin(mat)

    mat = mat/np.amax(mat)*255

    fig, figure_axes = plt.subplots( 1 , 1 )
    tracker = IndexTracker( figure_axes , mat ) 
    fig.canvas.mpl_connect( 'scroll_event' , tracker.onscroll )
    
    mng = plt.get_current_fig_manager()
    
    # mng.window.state('zoomed')
    mng.window.showMaximized()

    plt.show() 



class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape

        # self.ind = self.slices//2
        self.ind = 0

        self.im = ax.imshow(self.X[:, :, self.ind] , cmap='gray',vmin=0,vmax=255)
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
