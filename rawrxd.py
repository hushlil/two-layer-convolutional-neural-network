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

    # Initialize the list for hidden nodes z1 to z4
    z = []
    for i in range(4):
        weighted_sum = dual(0.0, 0.0)  # Initialize weighted sum for the hidden node

        # Apply w1 and w2 weights to the inputs as per the diagram
        if i == 0:  # Hidden node z1 (x1 -> z1, x2 -> z1)
            weighted_sum += w1 * x_dual[0] + w2 * x_dual[1]
        elif i == 1:  # Hidden node z2 (x2 -> z2, x3 -> z2)
            weighted_sum += w1 * x_dual[1] + w2 * x_dual[2]
        elif i == 2:  # Hidden node z3 (x3 -> z3, x4 -> z3)
            weighted_sum += w1 * x_dual[2] + w2 * x_dual[3]
        else:  # Hidden node z4 (x4 -> z4, x5 -> z4)
            weighted_sum += w1 * x_dual[3] + w2 * x_dual[4]

            # Apply ReLU activation (max(0, weighted_sum))
        zi = max(weighted_sum, dual(0.0, 0.0))  # ReLU activation function
        z.append(zi)  # Append to the hidden layer

    y = v1 * z[0] + v2 * z[1] + v3 * z[2] + v4 * z[3]

    return y, z

x_sample = [0.3, -1.5, 0.7, 2.1, 0.1]  # Sample input from question
y, z = convnet(x_sample)

print(f"Output y: Value = {y.val:.3f}, Gradient = {y.grad:.3f}")
print("Hidden layer z values (with gradients):")
for i, zi in enumerate(z):
    print(f"  z{i+1}: Value = {zi.val:.3f}, Gradient = {zi.grad:.3f}")
