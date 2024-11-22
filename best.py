"""
Implement two-layer convolutional neural network
 - x: List of 5 input values
 - Returns the output y and hidden nodes for z
"""
from ad import dual, max


def convnet(x):
    w1, w2 = dual(1.2, 1.0), dual(-0.2, 0.0)
    v1, v2, v3, v4 = dual(-0.3, 0.0), dual(0.6, 0.0), dual(1.3, 0.0), dual(-1.5, 0.0)

    # Converting inputs to dual numbers
    x_dual = [dual(xi, 1.0 if i == 0 else 0.0) for i, xi in enumerate(x)]

    z = []
    for i in range(4):
        weighted_sum = dual(0.0, 0.0)  # Initialise weighted sum for hidden node

        # Apply w1 and w2 weights to the inputs as per the diagram
        if i == 0:  # Hidden node z1 (x1 -> z1, x2 -> z1)
            weighted_sum += w1 * x_dual[0] + w2 * x_dual[1]
        elif i == 1:  # Hidden node z2 (x2 -> z2, x3 -> z2)
            weighted_sum += w1 * x_dual[1] + w2 * x_dual[2]
        elif i == 2:  # Hidden node z3 (x3 -> z3, x4 -> z3)
            weighted_sum += w1 * x_dual[2] + w2 * x_dual[3]
        else:  # Hidden node z4 (x4 -> z4, x5 -> z4)
            weighted_sum += w1 * x_dual[3] + w2 * x_dual[4]

        # ReLU activation
        zi = max(weighted_sum, dual(0.0, 0.0))
        z.append(zi)  # appending to keep hidden

    y = v1 * z[0] + v2 * z[1] + v3 * z[2] + v4 * z[3]

    return y, z


def get_user_input():
    user_input = input("Please enter exactly 5 numerical values separated by commas. For example: 0.3, -1.5, 0.7, 2.1, 0.1\n")

    if user_input:
        try:
            x_custom = [float(i.strip()) for i in user_input.split(',')]

            # only need 5 numbers for x1-x5
            if len(x_custom) != 5:
                print("Error: You must enter exactly 5 values.")
                return None  # Return None to indicate invalid input

            return x_custom
        except ValueError:
            print("Error: Invalid input. Please enter only numerical values separated by commas.")
            return None  # Return None to indicate invalid input
    else:
        return None  # Return None to indicate no input

x_sample = [0.3, -1.5, 0.7, 2.1, 0.1]  #  Sample input from question

x_custom = get_user_input()  # user input because question said any 5 numbers
if x_custom:
    y, z = convnet(x_custom)  # If user input is valid, use it
else:
    # if input is invalid, sample input is printed
    print(f"Invalid input or no input provided, using default sample input: {x_sample}")
    y, z = convnet(x_sample)

# Output the results
print(f"Output y: Value = {y.val:.3f}, Gradient = {y.grad:.3f}")
print("Hidden layer z values (with gradients):")
for i, zi in enumerate(z):
    print(f"  z{i + 1}: Value = {zi.val:.3f}, Gradient = {zi.grad:.3f}")