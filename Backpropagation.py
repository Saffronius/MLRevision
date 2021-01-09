import math


def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    return y

x1 = [0, 0,1 , 1]  
x2 = [0, 1, 0, 1]  
t = [0, 1, 1, 0]  

b1 = -0.3
w11 = 0.21
w21 = 0.10

b2 = 0.25
w12 = -0.4
w22 = 0.1

b3 = -0.4
w13 = -0.2
w23 = 0.3

error = 0
iteration = 0
train = True

while (train):

    for i in range(len(x1)):

        z_in1 = b1 + x1[i] * w11 + x2[i] * w21
        z_in2 = b2 + x1[i] * w12 + x2[i] * w22

        z1 = round(sigmoid(z_in1), 4)
        z2 = round(sigmoid(z_in2), 4)

        y_in = b3 + z1 * w13 + z2 * w23
        y = round(sigmoid(y_in), 4)

        del_k = round((t[i] - y) * y * (1 - y), 4)
        error = del_k
        
        w13 = round(w13 + del_k * z1, 4)
        w23 = round(w23 + del_k * z2, 4)
        b3 = round(b3 + del_k, 4)   

        del_1 = del_k * w13 * z1 * (1 - z1)
        del_2 = del_k * w23 * z2 * (1 - z2)

        b1 = round(b1 + del_1, 4)
        w11 = round(w11 + del_1 * x1[i], 4)
        w12 = round(w12 + del_1 * x1[i], 4)

        b2 = round(b2 + del_2, 4)
        w21 = round(w21 + del_2 * x2[i], 4)
        w22 = round(w22 + del_2 * x2[i], 4)

        print("Iteration: ", iteration)
        print("w11 : %5.4f w12: %5.4f w21: %5.4f w22: %5.4f w13: %5.4f  w23: %5.4f " % (w11, w12, w21, w22, w13, w23))
        print("Error: %5.3f" % del_k)
        iteration = iteration + 1

    if (iteration == 5000):
        train = False
