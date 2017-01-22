from numpy import *

def compute_error(initial_b, initial_m, points):
    total_error = 0.0
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]

        y_pred = initial_m * x + initial_b

        total_error += (y - y_pred) ** 2

    return total_error/float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    # Performing gradient descent
    for i in range(num_iterations):
        # update b and m with better and more accurate values by performing the gradient step
        b,m = gradient_step(b,m, array(points), learning_rate)
    return [b,m]

def gradient_step(current_b,current_m, points, learning_rate):

    # starting points for the gradients
    b_gradient = 0
    m_gradient = 0
    N = len(points)

    for i in range(0, N):
        x = points[i,0]
        y = points[i,1]

        # Computing the partial derivatives of the error function to find the direction to take the step in for b and m
        b_gradient += -(2/N) * (y - (current_m * x + current_b))
        m_gradient += -(2/N) * x * (y - (current_m*x +current_b))

    new_b = current_b - learning_rate * b_gradient
    new_m = current_m - learning_rate * m_gradient
    return [new_b,new_m]

def run():

    # Step 1 would be input the data
    points = genfromtxt('/Users/naveemoh/GitHub/ML_Algo_implementation/LinearRegression/data/data.csv',delimiter=",")

    # Step 2 define hyperparameters
    learning_rate=0.0001
    # initial b and m for eqn y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # Step 3 Train the model
    print('Starting gradient descent at b ={0} and m={1} and error = {2}'.format(initial_b,initial_m, compute_error(initial_b,initial_m,points)))
    # Gradient descent to get the final b and m vals
    [b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print('After {0} iterations b ={1} and m={2} and error = {3}'.format(num_iterations,b,m, compute_error(b,m,points)))




if __name__ == '__main__':
    run()
