# # Inputs for AND gate
# a = [0, 0, 1, 1]
# b = [0, 1, 0, 1]
# labels = [0, 0, 0, 1]

# # Initialize weights
# w1 = 0
# w2 = 0
# learning_rate = 0.1


# # Activation function
# def activation_function(weighted_sum):
#     return 1 if weighted_sum > 0 else 0


# # Training the perceptron
# while True:
#     errors = 0
#     for i in range(len(a)):
#         # Calculate weighted sum
#         weighted_sum = w1 * a[i] + w2 * b[i]

#         # Predict output
#         prediction = activation_function(weighted_sum)

#         # Calculate error
#         error = labels[i] - prediction

#         # Update weights
#         w1 += learning_rate * error * a[i]
#         w2 += learning_rate * error * b[i]

#         # Count errors
#         if error != 0:
#             errors += 1

#     # Stop training if there are no errors
#     if errors == 0:
#         break

# # Testing the perceptron
# print("Testing Perceptron for AND gate:")
# for i in range(len(a)):
#     weighted_sum = w1 * a[i] + w2 * b[i]
#     prediction = activation_function(weighted_sum)
#     print(f"Inputs: ({a[i]}, {b[i]}), Output: {prediction}")


def activation(val, thres):
    if val > thres:
        return 1
    else:
        return 0


def percepton(p, q):
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]
    y = [0, 0, 0, 1]
    w = [0.2, 1.7]
    lr = 0.1
    i = 0
    threshold = 1
    while i < 4:
        yo = a[i] * w[0] + b[i] * w[1]
        o = activation(yo, threshold)
        if o != y[i]:
            w[0] += lr * (y[i] - o) * a[i]
            w[1] += lr * (y[i] - o) * b[i]
            i = -1
        i += 1
    print("Final weights: ", w[0], w[1])
    return activation(p * w[0] + q * w[1], threshold)


print(percepton(0, 1))
