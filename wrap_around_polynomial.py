import ransac.core as ransac
import numpy as np

class WrapAroundPolynomial(ransac.Model):
    def __init__(self):
        super().__init__()
        self.degree = 0
        self.coefficients = []

    def Evaluate(self, x):  # Take an input variable x and return an output variable y
        if x < 0 or x > 1:
            raise ValueError("WrapAroundPolynomial.Evaluate(): The input x ({}) is expected to be in the [0, 1] range.".format(x))
        sum = self.coefficients[0]
        for power in range(1, self.degree + 1):
            sum += self.coefficients[power] * x**power
        return sum

    def Distance(self, y1, y2):  # Compute the distance between two output variables. Must return a float.
        return abs(y1 - y2)

    def Create(self, xy_tuples, **kwargs):  # Create a model with the given (x, y) observations
        self.degree = kwargs['degree']
        self.coefficients = []
        for coef_rank in range(self.degree + 1):
            self.coefficients.append(0)
        penultimateLast_coefs = [(0, 0)]
        for column in range(1, self.degree - 1):
            penultimate_coef = self.PenultimatePowerSolvingCoefficient(self.degree, column)
            last_coef = self.LastPowerSolvingCoefficient(self.degree, column)
            penultimateLast_coefs.append((penultimate_coef, last_coef))
        print ("WrapAroundPolynomial.Create(): penultimateLast_coefs = {}".format(penultimateLast_coefs))
        penultimate_power = self.degree - 1
        last_power = self.degree
        A = np.zeros((len(xy_tuples), self.degree - 1), dtype=float)
        b = np.zeros(len(xy_tuples), dtype=float)
        for row in range(len(xy_tuples)):
            (x, y) = xy_tuples[row]
            if x < 0 or x > 1:
                raise ValueError("WrapAroundPolynomial.Create(): x ({}) is expected to be in the [0, 1] range".format(x))
            A[row, 0] = 1.
            for column in range(1, self.degree - 1):
                A[row, column] += x**column
                penultimate_coef = penultimateLast_coefs[column][0]
                last_coef = penultimateLast_coefs[column][1]
                A[row, column] += penultimate_coef * x**penultimate_power + last_coef * x**last_power
            b[row] = y
        # A @ z = b
        z, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        print ("WrapAroundPolynomial.Create(): z = {}".format(z))
        for coef_rank in range(z.shape[0]):
            self.coefficients[coef_rank] = z[coef_rank]
        # Penultimate power coefficient
        penultimate_power_coef = 0
        for coef_rank in range(1, self.degree - 1):
            penultimate_power_coef += self.PenultimatePowerEvaluationCoefficient(self.degree, coef_rank) * self.coefficients[coef_rank]
        self.coefficients[-2] = penultimate_power_coef

        # Last power coefficient
        last_power_coef = 0
        for coef_rank in range(1, self.degree - 1):
            last_power_coef += self.LastPowerEvaluationCoefficient(self.degree, coef_rank) * self.coefficients[coef_rank]
        self.coefficients[-1] = last_power_coef


    def MinimumNumberOfDataToDefineModel(self, **kwargs):  # The minimum number of (x, y) observations to define the model
        return kwargs['degree'] - 1

    def PenultimatePowerSolvingCoefficient(self, degree, column):
        if column < 1 or column > degree - 2:
            raise ValueError("WrapAroundPolynomial.PenultimatePowerCoefficient(): column ({}) is expected to be in the range [1, {}]".format(column, degree - 2))
        if degree == 3 and column == 1:
            return -3
        if column == degree - 2:
            return -2
        return self.PenultimatePowerSolvingCoefficient(degree - 1, column) - 1

    def LastPowerSolvingCoefficient(self, degree, column):
        if column < 1 or column > degree - 2:
            raise ValueError("WrapAroundPolynomial.LastPowerCoefficient(): column ({}) is expected to be in the range [1, {}]".format(column, degree - 2))
        if degree == 3 and column == 1:
            return 2
        if column == degree - 2:
            return 1
        return self.LastPowerSolvingCoefficient(degree - 1, column) + 1

    def PenultimatePowerEvaluationCoefficient(self, degree, coef_rank):
        if degree == 3 and coef_rank == 1:
            return -3
        if coef_rank == degree - 2:
            return -2
        return self.PenultimatePowerEvaluationCoefficient(degree - 1, coef_rank) - 1

    def LastPowerEvaluationCoefficient(self, degree, coef_rank):
        if degree == 3 and coef_rank == 1:
            return 2
        if coef_rank == degree - 2:
            return 1
        return self.LastPowerEvaluationCoefficient(degree - 1, coef_rank) + 1

