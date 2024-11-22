#1
"""
from ad import dual  # Import the module as per the instructions

def convnet(x):
    # Given parameters
    w1, w2 = 1.2, -0.2
    v1, v2, v3, v4 = -0.3, 0.6, 1.3, -1.5

    # Ensure inputs are the correct type
    assert len(x) == 5, "Input list must have exactly 5 elements."

    # Hidden layer computations
    z = []
    for i in range(4):
        zi = max(0, w1 * x[i] + w2)  # ReLU activation
        z.append(dual(zi))  # Convert to dual type

    # Output computation
    y = v1 * z[0] + v2 * z[1] + v3 * z[2] + v4 * z[3]
    y = dual(y)  # Convert to dual type

    return y, z

x_sample = [0.3, -1.5, 0.7, 2.1, 0.1]
y, z = convnet(x_sample)
print(f"Output y: {y}")
print(f"Hidden layer z: {z}")
"""



#2
"""
from ad import dual  # Assuming ad.py is available

def relu(u):
    #ReLU activation for dual numbers.
    u_value, u_derivative = u
    if u_value >= 0:
        return u_value, u_derivative
    else:
        return 0, 0

def convnet(x):
    #Implement a two-layer CNN using dual numbers.
    # Parameters
    w1, w2 = 1.2, -0.2
    v1, v2, v3, v4 = -0.3, 0.6, 1.3, -1.5

    # Convert inputs to dual numbers
    x_dual = [dual(xi, 1 if i == 0 else 0) for i, xi in enumerate(x)]

    # Hidden layer calculations
    z = []
    for i in range(4):  # Four hidden nodes
        zi = relu((w1, 1) * x_dual[i] + (w2, 0))
        z.append(zi)

    # Output layer calculation
    y = (v1, 0) * z[0] + (v2, 0) * z[1] + (v3, 0) * z[2] + (v4, 0) * z[3]

    return y, z

x_sample = [0.3, -1.5, 0.7, 2.1, 0.1]
y, z = convnet(x_sample)
print("Output y:", y)
print("Hidden layer z:", z)

"""

#3
"""
from ad import dual, max

def convnet(x):

    # Parameters
    w1, w2 = dual(1.2, 0.0), dual(-0.2, 0.0)  # w1 and w2 are weights for hidden layer
    v1, v2, v3, v4 = dual(-0.3, 0.0), dual(0.6, 0.0), dual(1.3, 0.0), dual(-1.5, 0.0)  # Output layer weights

    # Ensure inputs are dual numbers
    x_dual = [dual(xi, 0.0) for xi in x]

    # Compute hidden layer outputs (z1 to z4)
    z = []
    for i in range(4):  # Four hidden nodes
        zi = max(w1 * x_dual[i] + w2, dual(0.0, 0.0))  # ReLU activation
        z.append(zi)

    # Compute output layer (scalar y)
    y = v1 * z[0] + v2 * z[1] + v3 * z[2] + v4 * z[3]

    return y, z
    
# Test the CNN implementation
x_sample = [0.3, -1.5, 0.7, 2.1, 0.1]
y, z = convnet(x_sample)

print(f"Output y: {y}")        # Scalar output
print(f"Hidden layer z: {z}")  # List of 4 hidden layer outputs

"""

"""
You need to implement a two-layer convolutional neural network (CNN) in Python that:

- Takes a list of 5 inputs [x1, x2, x3, x4, x5].
- Outputs:
  - A scalar value y of type dual.
  - A list [z1, z2, z3, z4] of 4 values of type dual.
  
Key details:
- Hidden layer: ReLU activation applied to 4 nodes.
- Output layer: A weighted sum of the 4 hidden nodes.
"""
from ad import dual, max

def convnet(x):
    # Define weights as dual numbers
    w1, w2 = dual(1.2, 1.0), dual(-0.2, 0.0)  # Ensure both w1 and w2 are dual numbers
    v1, v2, v3, v4 = dual(-0.3, 0.0), dual(0.6, 0.0), dual(1.3, 0.0), dual(-1.5, 0.0)
    # Convert inputs to dual numbers
    x_dual = [dual(xi, 0.0) for xi in x]  # Gradient of x_i is not tracked

    # Compute hidden layer nodes
    z = []  # Initialize list of hidden nodes
    for i in range(4):  # Loop over each hidden node z1, z2, z3, z4
        # Initialize weighted_sum as a dual number
        weighted_sum = dual(0.0, 0.0)
        for j in range(5):
            weighted_sum += w1 * x_dual[j] + w2  # Weighted sum from all inputs
        zi = max(weighted_sum, dual(0.0, 0.0))  # Apply ReLU activation
        z.append(zi)  # Append the result to z

# Compute output y as weighted sum of hidden nodes
    y = v1 * z[0] + v2 * z[1] + v3 * z[2] + v4 * z[3]
    # Return output y and hidden nodes z
    return y, z


# Example usage with a sample input
x_sample = [0.3, -1.5, 0.7, 2.1, 0.1]
y, z = convnet(x_sample)

# Display the output and hidden nodes
print(f"Output y: {y}")
print(f"Hidden layer z: {z}")




"""
    Two-layer CNN using dual numbers.
    Args:
          x (list): List of 5 input values.
      Returns:
          tuple: Scalar output `y` (dual) and list of 4 hidden layer outputs `z` (duals).
      """
