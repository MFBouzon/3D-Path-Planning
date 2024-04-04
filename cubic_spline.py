import sys
sys.path.append('../src')

import extra
import PathFillingPoints.Splines.Cubic3D as pfp
import PathFillingPoints.Splines.Cubic3DSolver as solver
import numpy as np


P=[[0,0,0],[1,1,1],[1,1,2],[1,2,2],[2,2,2]];


################################################################################
beta=0.0;
spl=pfp.CubicSpline3D(P,beta=beta);

print('\n')
print('    beta:',beta)
print('  MSE[0]:',spl.get_mse()[0])
print(' MSE[-1]:',spl.get_mse()[-1])
print('len(MSE):',len(spl.get_mse()))

print('min curvature:',np.min(spl.get_curvatures()));
print('max curvature:',np.max(spl.get_curvatures()));

extra.plot_data(P,spl,L=64);

################################################################################
beta=0.01;
spl=pfp.CubicSpline3D(P,beta=beta);

print('\n')
print('    beta:',beta)
print('  MSE[0]:',spl.get_mse()[0])
print(' MSE[-1]:',spl.get_mse()[-1])
print('len(MSE):',len(spl.get_mse()))

print('min curvature:',np.min(spl.get_curvatures()));
print('max curvature:',np.max(spl.get_curvatures()));

extra.plot_data(P,spl,L=64);