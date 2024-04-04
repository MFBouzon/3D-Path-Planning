from ursina import *
import numpy as np
import extra
import PathFillingPoints.Splines.Cubic3D as pfp
import PathFillingPoints.Splines.Cubic3DSolver as solver
from numpy.random import seed
from numpy.random import rand
from scipy.interpolate import interp1d
import math

app = Ursina()

class ParametricCurve(Entity):
    def __init__(self, equation, theta, t_range=(0, 2 * math.pi), segments=100, color=color.white):
        super().__init__()
        self.vertices=[]
        self.equation = equation
        self.t_range = t_range
        self.segments = segments
        self.theta = theta
        self.generate_curve()
        self.color = color
        self.model = Mesh(vertices=self.vertices, mode='line')

    def generate_curve(self):
        for i in range(self.segments + 1):
            t = lerp(self.t_range[0], self.t_range[1], i / self.segments)
            x, y, z = self.equation(t)
            self.vertices.append((x, y, z))
            
def parametric_equation(t):
    x = math.sin(t)
    y = math.cos(t)
    z = t / (2 * math.pi)
    return x, y, z

def polynomial(theta, t):
   
    
    print(theta)
    x = theta[0] + theta[1]*t + theta[2]*t*t + theta[3]*t*t*t
    y = theta[4] + theta[5]*t + theta[6]*t*t + theta[7]*t*t*t
    z = theta[8] + theta[9]*t + theta[10]*t*t + theta[11]*t*t*t

    return x,y,z


theta=rand(12);
#curve = ParametricCurve(polynomial, theta, t_range=(0,  1.2), segments=100)
#curve = ParametricCurve(parametric_equation, theta, t_range=(0,  math.pi/2), segments=100)

# Define the two 3D points
point1 = np.array([0, 0, 0])
point2 = np.array([3, 1, 2])
point3 = np.array([5, 0.5, 2])

# Define the degree of the parametric curve
N = 2

# Define the parameter t from 0 to 1
t = np.linspace(0, 1, 100)

# Define the parametric equation for the curve
def parametric_curve(t, coefficients, point1, point2, N):
    t = t[:, np.newaxis]
    powers = np.array([t ** i for i in range(N, -1, -1)])
    curve = np.dot(powers.T, coefficients)
    return (1 - t) * point1 + t * point2 + t * (1 - t) * curve


def parametric_curve2(t, coefficients, point1, point2, point3, N):
    t = t[:, np.newaxis]
    powers = np.array([t ** i for i in range(N, -1, -1)])
    curve = np.dot(powers.T, coefficients)
    return (1 - t) ** 2 * point1 + 2 * (1 - t) * t * point2 + t ** 2 * point3 + t * (1 - t) * curve


for i in range(10):
    # Generate random coefficients for the parametric equation
    coefficients = np.random.rand(N + 1, 3)
    # Generate the curve
    curve = parametric_curve(t, coefficients, point1, point2, N)

    #curve = interp1d([0,1], np.vstack([P1,P2]),axis=0)(t)
    #curve = np.array([P1 + (P2 - P1) * i for i in t])
    #print(curve[0])

    points = Entity(model=Mesh(vertices=curve[0].tolist(), mode='line', thickness=1.0), color=color.random)


for i in range(10):
    # Generate random coefficients for the parametric equation
    coefficients = np.random.rand(N + 1, 3)
    # Generate the curve
    curve = parametric_curve2(t, coefficients, point1, point2, point3, N)

    #curve = interp1d([0,1], np.vstack([P1,P2]),axis=0)(t)
    #curve = np.array([P1 + (P2 - P1) * i for i in t])
    #print(curve[0])

    points = Entity(model=Mesh(vertices=curve[0].tolist(), mode='line', thickness=1.0), color=color.random)

P = [point1, point2, point3]
spl = pfp.CubicSpline3D(P,beta=0.001)

print('  MSE[0]:',spl.get_mse()[0])
print(' MSE[-1]:',spl.get_mse()[-1])
print('len(MSE):',len(spl.get_mse()))

tline=np.linspace(0,2,100);
    
xline=np.zeros(100); 
yline=np.zeros(100);
zline=np.zeros(100);

curve = []

for l in range(len(tline)):
    r=spl.eval(tline[l]);
    xline[l]=r[0];
    yline[l]=r[1];
    zline[l]=r[2];
    curve.append((xline[l],yline[l],zline[l]))

points = Entity(model=Mesh(vertices=curve, mode='line', thickness=1.0), color=color.yellow)

points = Entity(model=Mesh(vertices=[point1,point2,point3], mode='point', thickness=.05), color=color.red)
EditorCamera()

app.run()