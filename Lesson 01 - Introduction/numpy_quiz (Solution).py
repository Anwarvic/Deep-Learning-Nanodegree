# Use the numpy library
import numpy as np


def prepare_inputs(inputs):
    # TODO: create a 2-dimensional ndarray from the given 1-dimensional list;
    #       assign it to input_array
    input_array = np.array(inputs).reshape(1, len(inputs))

    # TODO: find the minimum value in input_array and subtract that
    #       value from all the elements of input_array. Store the
    #       result in inputs_minus_min
    inputs_minus_min = input_array - np.min(input_array)

    # TODO: find the maximum value in inputs_minus_min and divide
    #       all of the values in inputs_minus_min by the maximum value.
    #       Store the results in inputs_div_max.
    inputs_div_max = inputs_minus_min / np.max(inputs_minus_min)

    # return the three arrays we've created
    return input_array, inputs_minus_min, inputs_div_max
    

def multiply_inputs(m1, m2):
    # TODO: Check the shapes of the matrices m1 and m2. 
    #       m1 and m2 will be ndarray objects.
    #
    #       Return False if the shapes cannot be used for matrix
    #       multiplication. You may not use a transpose
    m1_shape = m1.shape
    m2_shape = m2.shape
    if len(m1_shape)!=2 or len(m2_shape)!=2:
        return False
    if m1_shape[1] != m2_shape[0] and m2_shape[1] != m1_shape[0]:
        return False

    # TODO: If you have not returned False, then calculate the matrix product
    #       of m1 and m2 and return it. Do not use a transpose,
    #       but you swap their order if necessary
    if m1_shape[1] == m2_shape[0]:
        return np.dot(m1, m2)
    else:
        return np.dot(m2, m1)
    

def find_mean(values):
    # TODO: Return the average of the values in the given Python list
    return np.mean(np.array((values)))


input_array, inputs_minus_min, inputs_div_max = prepare_inputs([-1,2,7])
print("Input as Array: {}".format(input_array))
print("Input minus min: {}".format(inputs_minus_min))
print("Input  Array: {}".format(inputs_div_max))

print("Multiply 1:\n{}".format(multiply_inputs(np.array([[1,2,3],[4,5,6]]), np.array([[1],[2],[3],[4]]))))
print("Multiply 2:\n{}".format(multiply_inputs(np.array([[1,2,3],[4,5,6]]), np.array([[1],[2],[3]]))))
print("Multiply 3:\n{}".format(multiply_inputs(np.array([[1,2,3],[4,5,6]]), np.array([[1,2]]))))

print("Mean == {}".format(find_mean([1,3,4])))