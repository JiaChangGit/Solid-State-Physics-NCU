import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np


def create_animation(xxts, xlim=None, ylim=None):
    # x: shape with (#time, #shift)
    fig, ax = plt.subplots(figsize=(12,2))
    
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(np.min(xxts)-0.5, np.max(xxts)+0.5)
        
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(-1, 1)
        
    scatter, = ax.plot([],[], 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.close()

    def animate(i):
        scatter.set_data(xxts[i,:], 0)
        return [scatter]

    return matplotlib.animation.FuncAnimation(fig, animate, frames=xxts.shape[0], blit=True)

