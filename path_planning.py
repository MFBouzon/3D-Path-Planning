
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import numpy as np
import math
import time
from queue import PriorityQueue

import extra
import PathFillingPoints.Splines.Cubic3D as pfp

app = Ursina()

# Define a Voxel class.
# By setting the parent to scene and the model to 'cube' it becomes a 3d button.

class Voxel(Button):
    def __init__(self, position=(0,0,0)):
        super().__init__(parent=scene,
            position=position,
            model='wireframe_cube',
            origin_y=0,
            texture='white_cube',
            color=color.color(0, 0, random.uniform(.9, 1.0)),
            highlight_color=color.lime
        )
        self.neighbors = []

    def is_closed(self):
        return self.color == color.rgba(255,1,1,25)
    
    def is_open(self):
        return self.color == color.rgba(1,255,1,25)
    
    def is_obstacle(self):
        return self.color == color.rgba(50,50,50,150)

    def is_start(self):
        return self.color == color.orange
    
    def is_end(self):
        return self.color == color.cyan
    
    def is_path(self):
        return self.color == color.rgba(200,1,200,150)

    def reset(self):
        self.model='wireframe_cube'
        self.color=color.white
        self.highlight_scale = 1
        self.highlight_color = color.lime

    def make_closed(self):
        self.animate_color(color.rgba(255,1,1,25), duration=0.2)
        self.model='cube'
    
    def make_open(self):
        self.animate_color(color.rgba(1,255,1,25), duration=0.2)
        self.model='cube'
        
    def make_obstacle(self):
        self.model='cube'
        self.color=color.rgba(50,50,50,150)

    def make_start(self):
        self.model='cube'
        self.color=color.orange
    
    def make_end(self):
        self.model='cube'
        self.color=color.cyan
    
    def make_path(self):
        self.animate_color(color.rgba(200,1,200,150), duration=1)
        self.model='cube'

    def stop_highlight(self):
        self.highlight_scale = 0

    def start_highlight(self):
        self.highlight_scale = 1
    
    def update_neighbors(self, grid):
        self.neighbors = []
        row = int(self.position.x)
        col = int(self.position.y)
        depth = int(self.position.z)
        total_rows = len(grid)
        if row < total_rows - 1 and not grid[row + 1][col][depth].is_obstacle(): # DOWN
            self.neighbors.append(grid[row + 1][col][depth])
        if row > 0 and not grid[row - 1][col][depth].is_obstacle(): # UP
            self.neighbors.append(grid[row - 1][col][depth])
        if col < total_rows - 1 and not grid[row][col + 1][depth].is_obstacle(): # RIGHT
            self.neighbors.append(grid[row][col + 1][depth])
        if col > 0 and not grid[row][col - 1][depth].is_obstacle(): # LEFT
            self.neighbors.append(grid[row][col - 1][depth])
        if depth < total_rows - 1 and not grid[row][col][depth + 1].is_obstacle(): # FRONT
            self.neighbors.append(grid[row][col][depth + 1])
        if depth > 0 and not grid[row][col][depth - 1].is_obstacle(): # BACK
            self.neighbors.append(grid[row][col][depth - 1])



def h(v1, v2):
    return np.linalg.norm(v1.position-v2.position)

def reconstruct_path(came_from, current, start):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
        if current != start and current != end:
            current.make_path()
    return path

def AStar(grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for col in row for node in col}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for col in row for node in col}
    f_score[start] = h(start, end)

    open_set_hash = {start}

    while not open_set.empty():
        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            path = reconstruct_path(came_from, end, start)
            return path
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor, end)
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    if(neighbor != end):
                        neighbor.make_open()

        if current != start:
            current.make_closed()

    return None


def parametric_curve(coefficients, point1, point2, N=2, L=100):
    t = np.linspace(0, 1, L)
    t = t[:, np.newaxis]
    powers = np.array([t ** i for i in range(N, -1, -1)])
    curve = np.dot(powers.T, coefficients)
    return (1 - t) * point1 + t * point2 + t * (1 - t) * curve

def pujaico_curve1(points,
                    alpha=0.01,
                    beta=0.02,
                    weight_pr=1.0,
                    weight_pp=6.0,
                    weight_dpdp=2.0,
                    weight_ddpddp=2.0,
                    func_offset=0.35,
                    L=100):
    
    spl = pfp.CubicSpline3D(points,alpha=alpha, beta=beta, weight_pr=weight_pr,
                        weight_pp=weight_pp,
                        weight_dpdp=weight_dpdp,
                        weight_ddpddp=weight_ddpddp,
                        func_offset=func_offset,
                        show=True)

    print('  MSE[0]:',spl.get_mse()[0])
    print(' MSE[-1]:',spl.get_mse()[-1])
    print('len(MSE):',len(spl.get_mse()))
    print('min curvature:',np.min(spl.get_curvatures()))
    print('max curvature:',np.max(spl.get_curvatures()))

    tline=np.linspace(0,len(points)-1,L)
        
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

    return curve

def input(key):

    global start
    global end
    global grid
    global started
    global path
    global dof
    global curve_list
    
    if key == 'left mouse down' and not started:
        hit_info = raycast(camera.world_position, camera.forward)
        if hit_info.hit:
            if not start and not mouse.hovered_entity.is_end():
                start = mouse.hovered_entity
                start.make_start()

            elif not end and mouse.hovered_entity != start:
                end = mouse.hovered_entity
                end.make_end()
    
            elif mouse.hovered_entity != end and mouse.hovered_entity != start:
                mouse.hovered_entity.make_obstacle()

    if key == 'right mouse down' and mouse.hovered_entity and not started:
        if mouse.hovered_entity.is_start():
            start = None
        if mouse.hovered_entity.is_end():
            end = None
        mouse.hovered_entity.reset()

    if key == 'enter':
        started = True
        for row in grid:
            for col in row:
                for node in col:
                    node.update_neighbors(grid)
        
        path = AStar(grid, start, end)
        
        for row in grid:
            for col in row:
                for node in col:
                    node.stop_highlight()

    if key == 'c':

        start = None
        end = None
        started = False

        for row in grid:
            for col in row:
                for node in col:
                    node.reset()
                    
        for curve in curve_list:
            curve.disable()
            del(curve)

    if key == 't' and started:
        for i in range(0, len(path)-1):
            coefficients = np.random.rand(dof+1, 3)
            curve = parametric_curve(coefficients, path[i].position, path[i+1].position, dof, len(path))
            if(len(curve[0].tolist())>1):
                curve_list.append(Entity(model=Mesh(vertices=curve[0].tolist(), mode='line', thickness=2.0), color=color.white))
    
    if key == 'i' and started:
        coefficients = np.random.rand(dof+1, 3)
        curve = parametric_curve(coefficients, start.position, end.position, dof, len(path))
        if(len(curve[0].tolist())>1):
            curve_list.append(Entity(model=Mesh(vertices=curve[0].tolist(), mode='line', thickness=2.0), color=color.white))

    if key == 'p' and started:
        path_points = [(pos.position.x,pos.position.y,pos.position.z) for pos in path]
        path_points.append((start.position.x,start.position.y,start.position.z))
        curve_list.append(Entity(model=Mesh(vertices=path_points, mode='line', thickness=2.0), color=color.white))
    
    if key == 'f' and started:
        path_points = [(pos.position.x,pos.position.y,pos.position.z) for pos in path]
        path_points.append((start.position.x,start.position.y,start.position.z))
        curve = pujaico_curve1(path_points)
        curve_list.append(Entity(model=Mesh(vertices=curve, mode='line', thickness=4.0), color=color.yellow))
    

def make_grid(size):
    grid = []
    for x in range(size):
        grid.append([])
        for y in range(size):
            grid[x].append([])
            for z in range(size):
                voxel = Voxel(position=(x,y,z))
                player.ignore_list.append(voxel)
                grid[x][y].append(voxel)

    return grid



def update():
    player.y += held_keys['space'] *0.1
    player.y -= held_keys['left control'] *0.1



player = FirstPersonController(gravity=0)


start = None
end = None

run = True
started = False
grid_size = 10

dof = 5
path = []

grid = make_grid(grid_size)
print(grid[0][0][0].position)
curve_list = []

app.run()