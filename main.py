import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Store mean and standard deviation for each feature
mean_mu = []
std_dev = []


def load_data(filename):
    data_file = pd.read_csv(filename, usecols=['AGE', 'FEMALE', 'LOS', 'APRDRG', 'TOTCHG'])

    data_file = data_file.iloc[:, [0, 1, 2, 4, 3]]

    data = np.array(data_file, dtype=float)

    plot_param(data[:, :4], data[:, -1])
    # X = df.iloc[:,[0,1,2,5]].values
    # y = df.iloc[:,-2].values
    # data = data[0:,:5]

    normalize(data)

    return data[0:, :4], data[0:, -1]


def plot_param(x, y):
    plt.xlabel('Length of Stay')
    plt.ylabel('TOTCHG')
    plt.plot(x[:, 2], y, 'r+')
    plt.show()


def normalize(data):
    for i in range(0, data.shape[1] - 1):
        data[:, i] = ((data[:, i] - np.mean(data[:, i])) / np.std(data[:, i]))
        mean_mu.append(np.mean(data[:, i]))
        std_dev.append(np.std(data[:, i]))


def hypo_func(x, theta):
    return np.matmul(x, theta)


def cost_function(x, y, theta):
    return ((hypo_func(x, theta) - y).T @ (hypo_func(x, theta) - y)) / (2 * y.shape[0])


def gradient_descent(x, y, theta, learning_rate=0.1, num_iteration=10):
    m = x.shape[0]

    J_all = []

    for _ in range(num_iteration):
        hypo_x = hypo_func(x, theta)
        cost_ = (1 / m) * (x.T @ (hypo_x - y))
        theta = theta - (learning_rate) * cost_
        J_all.append(cost_function(x, y, theta))

    return theta, J_all


def plot_cost(J_all, num_iterations):
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.plot(num_iterations, J_all, 'm', linewidth="4")
    plt.show()


def test(theta, x):
    x[0] = (x[0] - mean_mu[0]) / std_dev[0]
    x[1] = (x[1] - mean_mu[1]) / std_dev[1]
    x[2] = (x[2] - mean_mu[2]) / std_dev[2]
    x[3] = (x[3] - mean_mu[3]) / std_dev[3]
    y = theta[0] + theta[1] * x[0] + theta[2] * x[1] + theta[3] * x[2] + theta[4] * x[3]

    print("Healthcare cost: ", y)


x, y = load_data("linear_regression_dataset.csv")
y = np.reshape(y, (500, 1))
x = np.hstack((np.ones((x.shape[0], 1)), x))

theta = np.zeros((x.shape[1], 1))
# print(theta)
learning_rate = 0.01
num_iterations = 200
theta, J_all = gradient_descent(x, y, theta, learning_rate, num_iterations)
J = cost_function(x, y, theta)
print("Cost: ", J)
print("Parameters: ", theta)

# for testing and plotting cost
n_iterations = []
jplot = []
count = 0
for i in J_all:
    jplot.append(i[0][0])
    n_iterations.append(count)
    count += 1
jplot = np.array(jplot)
n_iterations = np.array(n_iterations)
plot_cost(jplot, n_iterations)

test(theta, [17, 1, 1, 758])
