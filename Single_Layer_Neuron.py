import numpy as np

#Sigmoid function
def sigmoid(x, dType=np.int8):
    return 1/(1+np.exp(-x))

def derivative(x1, dType=np.int8):
    return x1 * (1 - x1)

"""
    #Played around with the sigmoid function
    sigmoid_test = np.array([1, 3, 1000])
    result = sigmoid(sigmoid_test, False)
    print([f"{num:.1000f}" for num in result])
"""

def if_user():
    print("The system will take a user input of a 4x3 matrix, as well as entering the desired result from each row.")
    
    #User input for expected input
    matrix_input = []
    for i in range(4):
        row_input = []
        for j in range(3):
            value = input(f'Please enter the value for row number {i+1} column number {j+1}: ')
            row_input.append(float(value)) 
        matrix_input.append(row_input)
    
    user_x = np.array(matrix_input, dtype=float)
    print(f'This is your resultant input:\n{user_x}')

    #User input for the expected output
    row_output = []
    
    for i in range(4):
        value = input(f'Please enter the output value for row number 1 column {i+1}: ')
        row_output.append([float(value)])  
    
    user_y = np.array(row_output)
    
    # Generate random weights between -1 and 1
    np.random.seed(32)
    random_weights = 2 * np.random.random((3, 1)) - 1

    times_ran = int(input("Input number of times you want the computer to run: "))

    for iter in range(times_ran):
        l0 = user_x
        l1 = sigmoid(np.dot(l0, random_weights))

        l1_error = user_y - l1
        l1_change = l1_error * derivative(l1)

        random_weights += np.dot(l0.T, l1_change)

    print(f'Final output:\n{l1}')



def demo():
    x = np.array([[0, 1, 0], 
                [0, 1, 1], 
                [0, 0, 1],
                [1, 1, 1]])
    # print(f'This is the input data to train on {x}')

    #Output, labelled result
    y = np.array([[0], [1], [1], [1]])

    print(f'This is the desired output {y}')

    np.random.seed(42)

    #Generate random weight
    random_weights = 2 * np.random.random((3,1)) -1

    #print(f'Initial weights{random_weights}')

    for iter in range(150000):
        l0 = x
        l1 = sigmoid(np.dot(l0, random_weights))

        l1_error = (y - l1) 
        l1_change = l1_error * derivative(l1)

        random_weights += np.dot(l0.T, l1_change)

    print(f'Final output {l1}')
    error_margin(l1, y)

def input_handle(user_input):
    if user_input.upper() == "DEMO":
        return demo()
    elif user_input.upper() == "USER":
        return if_user()
    else:
        print("Invalid input. Please enter 'demo' or 'user'.")
        new_input = input("Enter again: ") 
        return input_handle(new_input)


def error_margin(l1, y):
    errors = []
    
    for i in range(len(y)): 
        y_val = float(y[i][0])
        layer_val = float(l1[i][0])

        if y_val != 0:  
            error = abs(y_val - layer_val) / layer_val * 100
        else:
            error = abs(y_val - layer_val) * 100  
        errors.append(error)

    print(f"Error percentages for each output(%): {errors}")
     

def main():
    user_input = input("This is a single layered neural network, enter user to input the data for processing, enter demo for a system demonstration: ")
    input_handle(user_input)
    
    
if __name__ == "__main__":
    main()