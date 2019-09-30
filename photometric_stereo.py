import os
import re
import math
import numpy as np
from math import *
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# get all the pgm files, return in a list
def get_files(dir_path):
    files = os.listdir(dir_path) # list all the files under the directory
    files.pop() # remove Ambiance
    files.pop(0) # remove WS_FTP.LOG
    files.pop(0) # remove *.info
    return files   


# read pgm file and return an image matrix
def read_pgm(file_path):
    
    file = open(file_path,'rb')
    pgm_type = file.readline()

    [width, height] = [int(i) for i in file.readline().split()]
    max_value = int(file.readline())


    mat = np.zeros((192, 168), dtype = 'float') 
    for i in range(height):
        for j in range(width):
            mat[i][j] = ord(file.read(1))

    return mat



def least_square(V,I):  
    
    g = np.linalg.lstsq(V, I, rcond=None)[0]

    #print("g=",g)
    return g



def calculate_V(file_name, S_I, K):
    pattern_A = 'A.*E'
    pattern_E = 'E.*\\.'

    # use re to match azimuth and elevation in file name 
    azimuth = re.search(pattern_A, file).group().lstrip('A').rstrip('E')
    elevation = re.search(pattern_E, file).group().lstrip('E').rstrip('.')

    # turn string into int
    if azimuth[0] == '+':
        azimuth = int(azimuth.lstrip('+'))
    elif azimuth[0] == '-':
        azimuth = -int(azimuth.lstrip('-'))

    if elevation[0] == '+':
        elevation = int(elevation.lstrip('+'))
    elif elevation[0] == '-':
        elevation = -int(elevation.lstrip('-'))


    z = cos(math.radians(azimuth))*cos(math.radians(elevation))
    x = sin(math.radians(elevation))
    y = -sin(math.radians(azimuth))*cos(math.radians(elevation))


    # magnitude = S_I
    x *= S_I
    y *= S_I
    z *= S_I
  
    # multiply scaling factor K
    x *= K
    y *= K
    z *= K

    return [x, y, z]




def get_face_2(N_list):

    # f_list stores the value of z at each point
    f_list = np.zeros((192, 168), dtype = 'float') 
    # albedo stores the value of albedo at each point
    albedo = np.zeros((192, 168, 3), dtype = 'float') 

    albedo[0][0] = sqrt(N_list[0][0][0]**2+N_list[0][0][1]**2+N_list[0][0][2]**2)

    max_albedo = 0 # track the max value of albedo

    # integral the first line
    for i in range(1,168):
        N = N_list[0][i]
        d_y = N[1]/N[2]
        f_list[0][i] = f_list[0][i-1]+d_y
        curr_albedo = sqrt(N[0]**2+N[1]**2+N[2]**2)


        if curr_albedo > max_albedo:
            max_albedo = curr_albedo

        albedo[0][i][0] = curr_albedo
        albedo[0][i][1] = curr_albedo
        albedo[0][i][2] = curr_albedo

    
    # integral rest of the surface
    for i in range(1,192):
        for j in range(168):
            N = N_list[i][j]
            d_x = N[0]/N[2]
            f_list[i][j] = f_list[i-1][j]+d_x
            curr_albedo = sqrt(N[0]**2+N[1]**2+N[2]**2)
            if curr_albedo > max_albedo:
                max_albedo = curr_albedo

            albedo[i][j][0] = curr_albedo
            albedo[i][j][1] = curr_albedo
            albedo[i][j][2] = curr_albedo  

    # normalize albedo to 0-1
    albedo = albedo/max_albedo
    
    return f_list, albedo



def get_face_1(N_list):
    f_list = np.zeros((192, 168), dtype = 'float') 
    albedo = np.zeros((192, 168, 3), dtype = 'float') 

    albedo[0][0] = sqrt(N_list[0][0][0]**2+N_list[0][0][1]**2+N_list[0][0][2]**2)

    max_albedo = 0
    for i in range(1,192):
        N = N_list[i][0]
        d_x = N[0]/N[2]
        f_list[i][0] = f_list[i-1][0]+d_x
        curr_albedo = sqrt(N[0]**2+N[1]**2+N[2]**2)
        if curr_albedo > max_albedo:
            max_albedo = curr_albedo

        albedo[i][0][0] = curr_albedo
        albedo[i][0][1] = curr_albedo
        albedo[i][0][2] = curr_albedo


    for j in range(1,168):
        for i in range(0,192):
            N = N_list[i][j]
            d_y = N[1]/N[2]
            f_list[i][j] = f_list[i][j-1]+d_y
            curr_albedo = sqrt(N[0]**2+N[1]**2+N[2]**2)
            if curr_albedo > max_albedo:
                max_albedo = curr_albedo

            albedo[i][j][0] = curr_albedo
            albedo[i][j][1] = curr_albedo
            albedo[i][j][2] = curr_albedo

    albedo = albedo/max_albedo
    
    return f_list, albedo





def plot_surface(f, albedo):

    x = np.arange(0, 168, 1)
    y = np.arange(0, 192, 1)
    X, Y = np.meshgrid(x, y)
    #print(X.shape, Y.shape, f.shape, albedo.shape)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    #print(X.shape, Y.shape, f.shape)
    ax.plot_surface(X,Y,f, rstride=1, cmap = 'gray', facecolors = albedo)
    #ax.scatter(x,y,f, c = albedo)
    #ax.plot_wireframe(X, Y, f, rstride=10, cstride=10, cmap = albedo) 
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('z', fontsize=18)

    plt.show()


def plot_scatter(f, albedo):

    fig = plt.figure()
    ax = Axes3D(fig)
    X = []
    Y = []
    Z = []
    A = []        
    for i in range(192):
        for j in range(168):
            X.append(i)
            Y.append(j)
            Z.append(f[i][j])
            A.append(albedo[i][j])

    
    ax.scatter(X,Y,Z,c=A,cmap='gray')
    ax.view_init(elev=40,azim=340)  #40 340
    
    
    plt.show()



def plot_heatmap(N_list):


    # normalize the normal vector
    Normalized_N = np.zeros((192, 168, 3), dtype = 'float')
    for i in range(192):
        for j in range(168):
            length = sqrt(N_list[i][j][0]**2+N_list[i][j][1]**2+N_list[i][j][2]**2)
            Normalized_N[i][j][0] = N_list[i][j][0]/length
            Normalized_N[i][j][1] = N_list[i][j][1]/length
            Normalized_N[i][j][2] = N_list[i][j][2]/length

    sns.heatmap(Normalized_N[:,:,0])
    plt.show()
    sns.heatmap(Normalized_N[:,:,1])
    plt.show()
    sns.heatmap(Normalized_N[:,:,2])
    plt.show()



def plot_albedo(albedo):
    plt.imshow(albedo)
    plt.show()


if __name__ == '__main__':


    S_I = 100 # fixed source intensity
    K = 1     # fixed scaling factor K


    dir_path = './CroppedYale/yaleB01/' # source directory of photos
    file_list = get_files(dir_path)     # get the list of files in the dir
    num_file = len(file_list)

    matrix_list = []
    V_matrix = []


    # calculate the source direction of each picture
    for file in file_list: 
        matrix_list.append(read_pgm(dir_path+file))
        V_matrix.append(calculate_V(file, S_I, K))

    V_matrix = np.array(V_matrix)


    # use least square regression to fit g(x,y) (albedo and normal vector)
    N_list = np.zeros((192, 168, 3), dtype = 'float')  # list of normal vector

    for i in range(192): # height = 192
        for j in range(168): # width = 168
            I_matrix = [[a[191-i][j]] for a in matrix_list]
            I_matrix = np.array(I_matrix)
            g = least_square(V_matrix,I_matrix)
            N_list[i][j] = [g[0][0], g[1][0], g[2][0]]


    # integral to get the whole surface
    #f, albedo = get_face_1(N_list)
    f, albedo = get_face_2(N_list)


    # plot scatter
    plot_scatter(f, albedo)

    # plot the surface
    #plot_surface(f, albedo)

    # plot heatmap
    plot_heatmap(N_list)

    # plot recoverd albedo
    plot_albedo(albedo)


