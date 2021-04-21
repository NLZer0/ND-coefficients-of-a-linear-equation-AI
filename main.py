import numpy as np

#27.34x^2 -9.992x + 192
def f(x):
    return 12.001 * x**3 + 1.34*x*x + 0*x + 11

train_size = 100
test_size = 50
h = 0.1
train_x = [0]*train_size
train_y = [0]*train_size
train_x[0] = -5
train_y[0] = f(train_x[0])


test_x = [0]*test_size
test_y = [0]*test_size
test_x[0] = 567
test_y[0] = f(train_x[0])

for i in range(1,train_size):
    train_x[i] = train_x[i-1]+h
    train_y[i] = f(train_x[i])
for i in range(1,test_size):
    test_x[i] = test_x[i-1]+h
    test_y[i] = f(test_x[i])

power_or_ec = 3
weigths = np.random.random(power_or_ec+1)
alpha = 0.0005

for it in range(500):
    sum_of_er = 0
    for i in range(train_size):
        lay_0 = np.array([])
        for j in range(power_or_ec+1):
            lay_0 = np.insert(lay_0, j, train_x[i]**j)
        lay_1 = lay_0.dot(weigths.T)

        sum_of_er = np.sum((lay_1 - train_y[i])**2)
        deltas = train_y[i] - lay_1 

        weigths += alpha * lay_0.dot(deltas)
    
    if (it % 10 == 9):
        print ("Error:" + str(sum_of_er)) 

print(weigths)