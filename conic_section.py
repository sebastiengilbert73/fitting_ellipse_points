import ransac.core as ransac
import numpy as np

class ConicSection(ransac.Model):  # Input is (x, y) and output is Ax**2 + Bxy + Cy**2 + Dx + Ey + F.
    #                                A point belonging to the conic section will return 0
    def __init__(self):
        super().__init__()
        self.A = 0
        self.B = 0
        self.C = 0
        self.D = 0
        self.E = 0
        self.F = 0

    def Evaluate(self, xy):  # Take an input variable x and return an output variable y
        x = xy[0]
        y = xy[1]
        return self.A * x**2 + self.B * x * y + self.C * y**2 + self.D * x + self.E * y + self.F

    def Distance(self, y1, y2):  # Compute the distance between two output variables. Must return a float.
        return abs(y1 - y2)

    def Create(self, xy_tuples, **kwargs):  # Create a model with the given (x, y) observations
        A = np.zeros((len(xy_tuples), 6), dtype=float)
        for obsNdx in range(len(xy_tuples)):
            ((x, y), d) = xy_tuples[obsNdx]
            A[obsNdx, 0] = x**2
            A[obsNdx, 1] = x * y
            A[obsNdx, 2] = y**2
            A[obsNdx, 3] = x
            A[obsNdx, 4] = y
            A[obsNdx, 5] = 1
        # Cf. https://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
        # Find the eigenvalues and eigenvector of A^T A
        e_vals, e_vecs = np.linalg.eig(np.dot(A.T, A))
        #print ("ConicSection.Create(): e_vals = {}".format(e_vals))
        #print ("ConicSection.Create(): e_vecs = {}".format(e_vecs))
        # Extract the eigenvector (column) associated with the minimum eigenvalue
        z = e_vecs[:, np.argmin(e_vals)]
        #print("ConicSection.Create(): z = {}".format(z))
        self.A = z[0]
        self.B = z[1]
        self.C = z[2]
        self.D = z[3]
        self.E = z[4]
        self.F = z[5]

    def MinimumNumberOfDataToDefineModel(self, **kwargs):  # The minimum number or (x, y) observations to define the model
        return 6

    def ConicSectionType(self, threshold=1.E-15):
        # Cf. https://www.varsitytutors.com/hotmath/hotmath_help/topics/conic-sections-and-standard-forms-of-equations
        gamma = self.B**2 - 4 * self.A * self.C
        print("ConicSectionType(): gamma = {}".format(gamma))
        if abs(gamma) <= threshold:
            return 'parabola'
        if gamma < 0:
            return 'ellipse'
        else:
            return 'hyperbola'