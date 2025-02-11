import numpy as np

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def demo():
    x = np.array([[3, 3, 5], 
                [22,55, 22], 
                [123, 123, 123],
                [0, 0, 0]])
    # print(f'This is the input data to train on {x}')

    #Output, labelled result
    y = np.array([[3], [22], [123], [0]])

    print(f'This is the desired output {y}')

    np.random.seed(42)

    #Generate random weight
    random_weights_first = 2 * np.random.random((3,1)) -1
    random_weights_second = 2 * np.random.random([3,1]) - 1
    

    #print(f'Initial weights{random_weights}')

    for iter in range(150000):
        l0 = x
        l1 = leaky_relu(np.dot(l0, random_weights_first))
        l2 = leaky_relu(np.dot(11, random_weights_second))

        l1_error = (y - l1) 
        l2_error = l1 - l2
        l2_change = l2_error * leaky_relu_derivative(np.dot(l0, random_weights_second))
        l1_change = l1_error * leaky_relu_derivative(np.dot(l0, random_weights_first))

        random_weights_first += np.dot(l0.T, l1_change)
        random_weights_second += np.dot(l1.T, l2_change)
        x.shape

    print(f'Final output {l1}')


def main():
    demo()

main()