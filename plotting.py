import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_benchmark_evaluations(evaluations, problem):

    #v_func = np.vectorize(problem.pareto_front)

    x = evaluations[:,0]
    y = evaluations[:,1]

    #x_pareto = np.linspace(problem.lower_bound, problem.upper_bound, len(x))
    #y_pareto = v_func(x_pareto)

    plt.scatter(x, y, c='blue')
    #plt.plot(x_pareto, y_pareto, c='black')
    plt.title(problem.get_name())
    plt.xlabel('f1')
    plt.ylabel('f2')
    
    plt.show()


def plot_evaluations(evaluations, problem):

    x = evaluations[:, 0]
    y = evaluations[:, 1]

    plt.scatter(x, y, c='blue')
    plt.title(problem.get_name())
    plt.xlabel('f1')
    plt.ylabel('f2')

    plt.show()

def plot_3d_evaluations(evaluations, problem):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    x = evaluations[:,0]
    y = evaluations[:,1]
    z = evaluations[:,2]

    ax.scatter(x,y,z, color='navy', marker='o')

    # final figure details
    ax.set_xlabel('$f_1()$', fontsize=15)
    ax.set_ylabel('$f_2()$', fontsize=15)
    ax.set_zlabel('$f_3()$', fontsize=15)
    ax.view_init(elev=11, azim=-21)
    plt.autoscale(tight=True)
    plt.savefig("test.svg", format="svg")
    plt.show()


def plot_reference_points(reference_points):
    x = []
    y= []
    for point in reference_points:
        x.append(point[0])
        y.append(point[1])

    plt.scatter(x,y)

    plt.show()

def plot_3d_reference_points(reference_points):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    x = []
    y = []
    z = []

    for point in reference_points:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    ax.scatter(x,y,z, color='navy', marker='o')

    # final figure details
    ax.set_xlabel('$f_1()$', fontsize=15)
    ax.set_ylabel('$f_2()$', fontsize=15)
    ax.set_zlabel('$f_3()$', fontsize=15)
    ax.view_init(elev=11, azim=-21)
    plt.autoscale(tight=True)
    plt.show()