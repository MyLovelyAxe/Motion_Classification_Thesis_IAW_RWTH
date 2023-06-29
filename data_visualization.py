import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime as dt
from itertools import count
from util.utils import get_prepared, calc_axis_limit, plot_func


####################################
###### get general parameters ######
####################################

parser = argparse.ArgumentParser(description='Visualize original csv data')

### general ###
# parser.add_argument('--data_path', type=str, default='dataset/chor2_20230609')
parser.add_argument('--data_path', type=str, default='dataset/testset_20230627')

### animation ###
parser.add_argument('--start_frame', type=int, default=3400, help='from which frame to start visualize')
parser.add_argument('--end_frame', type=int, default=4600, help='to which frame to end visualize')
parser.add_argument('--output_anim', type=bool, default=False, help='whether to output animation of visualization')

args = parser.parse_args([])

if __name__ == '__main__':

    input_path = os.path.join(args.data_path,'unknown.NoHead.csv')
    coords,dist_time,joints_dict = get_prepared(input_path,frame_range=[args.start_frame,args.end_frame])
    N_frames = args.end_frame - args.start_frame

    ##################
    ###### plot ######
    ##################

    fig = plt.figure(figsize=(16,8))
    fig.tight_layout()

    ### ax1: original coordinates in 3D ###
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.view_init(30, 150)
    high_1, low_1 = calc_axis_limit(coords) # (x_high, x_low), (y_high, y_low), (z_high, z_low)
    title_1 = f'Original coordinates in frames {args.start_frame}-{args.end_frame}'

    #### ax2: distance changing in 2D ###
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(N_frames)
    dist_plots = [ax2.plot(x, dist_time[idx],label=f'{links}')[0] for idx,(links,joints) in enumerate(joints_dict.items())]
    plt.legend()
    title_2 = f'Distances of limbs in frames {args.start_frame}-{args.end_frame}'

    ### animation ###
    ani1 = animation.FuncAnimation(fig,plot_func,frames=N_frames,fargs=(ax1,joints_dict,coords,high_1,low_1,title_1,
                                                                        ax2,dist_plots,dist_time,title_2),interval=17)
    plt.show()

    #### save animation.gif ###
    if args.output_anim:
        writergif = animation.PillowWriter(fps=30) 
        ani1.save(os.path.join(args.data_path,f"anim_{dt.now().strftime('%d-%h-%H-%M')}_{args.data_path.split('/')[1]}.gif"), writer=writergif)
