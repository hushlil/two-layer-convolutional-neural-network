"""
Implement two-layer convolutional neural network
 - x: List of 5 input values
 - Returns the output y and hidden nodes for z
"""

from ad import dual, max

def convnet(x):
    # Define weights as dual numbers
    w1, w2 = dual(1.2, 1.0), dual(-0.2, 0.0)
    v1, v2, v3, v4 = dual(-0.3, 0.0), dual(0.6, 0.0), dual(1.3, 0.0), dual(-1.5, 0.0)

    x_dual = [dual(xi, 1.0 if i == 0 else 0.0) for i, xi in enumerate(x)]  # converting inputs to dual numbers

    # my hidden layer nodes
    z = []  # initialising the list for z1 to z4
    for i in range(4):
        weighted_sum = dual(0.0, 0.0)  # Initialise weighted sum as a dual number

        if i == 0:
            weighted_sum += w1 * x_dual[0] + w2 * x_dual[1]  # For z1 (x1 -> z1, x2 -> z1)
        elif i == 1:
            weighted_sum += w1 * x_dual[1] + w2 * x_dual[2]  # For z2 (x2 -> z2, x3 -> z2)
        elif i == 2:
            weighted_sum += w1 * x_dual[2] + w2 * x_dual[3]  # For z3 (x3 -> z3, x4 -> z3)
        else:
            weighted_sum += w1 * x_dual[3] + w2 * x_dual[4]  # For z4 (x4 -> z4, x5 -> z4)

        # ReLU activation time
        zi = max(weighted_sum, dual(0.0, 0.0))
        z.append(zi) #appending to keep hidden

    y = v1 * z[0] + v2 * z[1] + v3 * z[2] + v4 * z[3]

    return y, z

x_sample = [0.3, -1.5, 0.7, 2.1, 0.1]  # Sample input from question
y, z = convnet(x_sample)

print(f"Output y: {y}")
print(f"Hidden layer z: {z}")
for i, zi in enumerate(z):
    print(f"  z{i+1}: Value = {zi.val}, Gradient = {zi.grad}")


