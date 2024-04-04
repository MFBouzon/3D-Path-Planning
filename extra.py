import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import numpy as np

def plot_data(P:list,spl,L=64):
    N=len(P);
    
    tline=np.linspace(0,N-1,L);
    
    xline=np.zeros(L); 
    yline=np.zeros(L);
    zline=np.zeros(L);

    for l in range(len(tline)):
        r=spl.eval(tline[l]);
        xline[l]=r[0];
        yline[l]=r[1];
        zline[l]=r[2];
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot3D(xline, yline, zline, 'gray')

    Pn=np.array(P);
    ax.scatter(Pn[:,0], Pn[:,1], Pn[:,2]);
    plt.show()
