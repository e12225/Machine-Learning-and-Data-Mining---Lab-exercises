import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import unittest

# My registration number = E/12/225 >> 225. Hence e1 = 2, e2 = 2, e3 = 5

matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('legend', fontsize=20)

x = np.linspace(-10, 10, 100)


class GradientDescentAlgorithm:
    # Plotting the behavior of f(x)
    def function(self):

        fig1 = plt.figure(figsize=(30, 30))
        axes1 = plt.subplot(1, 1, 1)
        fig1.text(0.5, 0.04, 'X', ha='center', fontsize=15)
        fig1.text(0.04, 0.5, 'f(X)', va='center', rotation='vertical', fontsize=15)

        y = (2 * pow(x, (2 % 5))) - (5 * pow(x, (2 % 5))) - (2 * pow(x, (5 % 5))) + 5
        # this f(x) function doesn't have a local minima.

        plt.ylim(-100, 10)
        axes1.plot(x, y, linewidth=.2, ls='--', color='b', marker=".")
        axes1.legend("f(x)", loc="upper right")
        axes1.set_title("f(x) Vs x")
        plt.savefig('f_x.png', dpi=400, bbox_inches='tight')

        dif_y = -3 * 2 * x  # differentiation of f(x)
        print("This function doesn't have a local minima. It has a local maxima")
        print("The local maximum of this function = " + str(np.max(y)))

        return dif_y

    # Gradient descent algorithm
    def algorithm(self, dif_y, current_x, learning_rate, precision):
        k = 0
        previous_step_size = current_x
        index_of_current_x = x.tolist().index(
            current_x)  # obtaining the index of the starting x point, within x[] array

        while previous_step_size > precision:

            if index_of_current_x == x.size:
                break

            current_x = x[index_of_current_x]
            previous_x = pow(current_x, k)

            dif_y_k = dif_y[np.isclose(x, current_x)]

            x_k = pow(current_x, k) - (learning_rate * dif_y_k)

            previous_step_size = x_k - previous_x

            if previous_step_size < 0:
                previous_step_size *= (-1)

            index_of_current_x += 1

        return current_x


'''Answers for Question 2, 3 and 4 are as follow
'''

# Question 2:
''' "current_x" is selected such that it is greater than "precision".
    Because, initially, "previous_step_size" is equal to "current_x" and the while loop will run only if 
    "previous_step_size" > "precision". So, in order to make sure it is proceeded to the algorithm, "current_x" value
    should be selected as it is greater than "precision".  
'''

# Question 3:
''' When changing the learning rate from a smaller gradient size to a larger gradient size, the value of pow(x, k)
    confidently moves in the direction of the negative gradient since it is recalculated frequently. But calculating 
    consumes more time. So it takes bit longer time (here it is negligible) to reach the end. 
'''


class GradientDescentAlgorithmTest(unittest.TestCase):
    def setUp(self):
        """Setting up for the test"""
        print("GradientDescentAlgorithm: setUp_:begin")
        self.gradientDescent = GradientDescentAlgorithm()

    def tearDown(self):
        """Cleaning up after the test"""
        print("GradientDescentAlgorithm: tearDown_:begin")

    def testAlgorithm(self):
        self.gradientDescent.function()
        ''' in order to validate, the return value of def function() and return value of def algorithm() 
        should not be equal '''
        self.assertNotEquals(self.gradientDescent.algorithm(self.gradientDescent.function(), x[2], x[5] - x[0], -1),
                             2.96939087848)


if __name__ == '__main__':
    unittest.main()
