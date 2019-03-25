import ssl
import numpy as np
from numpy import linalg as LA
import math
from sklearn import preprocessing
import glob
import functools
from MapFileUpload import MapFileUpload


def random_unit_vector():
    phi = 2.0 * math.pi * np.random.random()
    z = 2.0 * np.random.random() - 1.0
    r = math.sqrt(1.0 - z * z)
    return np.array([r * math.cos(phi), r * math.sin(phi), z])

class Rotation:
    """
    * Rotation : provides a representation for 3D space rotations
    * using euler angles (ZX'Z'' convention) or rotation matrices
    """

    def _euler2mat_z1x2z3(self, z1=0, x2=0, z3=0):
        cosz1 = math.cos(z1)
        sinz1 = math.sin(z1)
        Z1 = np.array(
            [[cosz1, -sinz1, 0],
             [sinz1, cosz1, 0],
             [0, 0, 1]])

        cosx = math.cos(x2)
        sinx = math.sin(x2)
        X2 = np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]])

        cosz3 = math.cos(z3)
        sinz3 = math.sin(z3)
        Z3 = np.array(
            [[cosz3, -sinz3, 0],
             [sinz3, cosz3, 0],
             [0, 0, 1]])

        return functools.reduce(np.dot, [Z1, X2, Z3])

    def _mat2euler(self, M):
        M = np.asarray(M)
        try:
            sy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            sy_thresh = _FLOAT_EPS_4
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        sy = math.sqrt(r31 * r31 + r32 * r32)
        if sy > sy_thresh:
            x2 = math.acos(r33)
            z1 = math.atan2(r13, -r23)
            z3 = math.atan2(r31, r32)
        else:
            x2 = 0
            z3 = 0
            z1 = math.atan2(r21, r22)
        return (z1, x2, z3)

    def _init_from_angles(self, z1, x2, z3):
        self._z1, self._x2, self._z3 = z1, x2, z3
        self._M = self._euler2mat_z1x2z3(self._z1, self._x2, self._z3)

    def _init_from_matrix(self, matrix):
        self._M = np.asarray(matrix)
        self._z1, self._x2, self._z3 = self._mat2euler(self._M)

    def __init__(self, arg1=None, x2=None, z3=None):
        if arg1 is None:
            self._init_from_angles(0, 0, 0)  # loads identity matrix
        elif x2 is not None:
            self._init_from_angles(arg1, x2, z3)
        elif arg1.size == 3:
            self._init_from_angles(arg1[0], arg1[1], arg1[2])
        else:
            self._init_from_matrix(arg1)

    def matrix(self, new_matrix=None):
        if new_matrix is not None:
            self._init_from_matrix(new_matrix)
        return self._M

    def euler_angles(self, z1=None, x2=None, z3=None):
        if z1 is not None:
            self._init_from_angles(z1, x2, z3)
        return (self._z1, self._x2, self._z3)

    def random(self):
        V = 2. * math.pi * np.random.random(), np.arccos(
            2.0 * np.random.random() - 1.0), 2. * math.pi * np.random.random()
        self.euler_angles(V)

class TripletHamiltonian:
    def __init__(self):
        self.Id = np.matrix('1 0 0; 0 1 0; 0 0 1', dtype=np.complex_)
        self.Sz = np.matrix('1 0 0; 0 0 0; 0 0 -1', dtype=np.complex_)
        self.Sx = np.matrix('0 1 0; 1 0 1; 0 1 0', dtype=np.complex_) / math.sqrt(2.0)
        self.Sy = - 1j * np.matrix('0 1 0; -1 0 1; 0 -1 0', dtype=np.complex_) / math.sqrt(2.0)

    def fine_structure(self, D, E, rotation=Rotation()):
        rotation_matrix = rotation.matrix()
        rSx = rotation_matrix[0, 0] * self.Sx + rotation_matrix[0, 1] * self.Sy + rotation_matrix[0, 2] * self.Sz
        rSy = rotation_matrix[1, 0] * self.Sx + rotation_matrix[1, 1] * self.Sy + rotation_matrix[1, 2] * self.Sz
        rSz = rotation_matrix[2, 0] * self.Sx + rotation_matrix[2, 1] * self.Sy + rotation_matrix[2, 2] * self.Sz
        return D * (np.dot(rSz, rSz) - 2. * self.Id / 3.) + E * (np.dot(rSy, rSy) - np.dot(rSx, rSx))

    def zeeman(self, Bx, By, Bz):
        return Bx * self.Sx + By * self.Sy + Bz * self.Sz

    def spin_hamiltonian_mol_basis(self, D, E, B, theta, phi):
        Bz = B * math.cos(theta)
        Bx = B * math.sin(theta) * math.cos(phi)
        By = B * math.sin(theta) * math.sin(phi)

        return self.fine_structure(D, E) + self.zeeman(Bx, By, Bz)

    def spin_hamiltonian_field_basis(self, D, E, B, theta, phi):
        return self.fine_structure(D, E, Rotation(0, -theta, -phi + math.pi / 2.)) + self.zeeman(0, 0, B)

    def eval(self, D, E, B, theta=0, phi=0, mol_basis=True):
        if mol_basis:
            return np.linalg.eigvalsh(self.spin_hamiltonian_mol_basis(D, E, B, theta, phi))
        else:
            return np.linalg.eigvalsh(self.spin_hamiltonian_field_basis(D, E, B, theta, phi))

def scoring(score, best_scores_list):
    least_best = best_scores_list[-1]
    if score > least_best:
        best_scores_list = np.delete(np.sort(np.append(best_scores_list, score)), 0)
    return best_scores_list




trp = TripletHamiltonian()
trp.D = 1414
trp.E = 10.3
trp.B = 0.
gauss2mhz = 0.002799

dA = 45.0
a = math.radians(90.0) * (1.0 / dA + 1.0)  # 91 degree for theta and phi
b = a / dA
Phi = np.arange(0, a, b)
Theta = np.arange(0, a, b)

best_scores_list = np.zeros((10, 1))
best_scores_angles = np.zeros((10, 3))

opsysWind = False
filenames = "megaavodmrB240518Long.dat", "megaavodmrlowB270518.dat", "megaavodmrzoomB060618.dat", "megaavodmrlowB120618.dat", "megaavodmrlocB140618.dat"
datapointsN, experiment_dict = MapFileUpload(filenames, opsysWind)

index_Phi = 0
for trp.phi in Phi:
    index_Theta = 0
    Phi_deg = round(float((trp.phi * 180) / math.pi))

    for trp.theta in Theta:
        Theta_deg = round(float((trp.theta * 180) / math.pi)) #converting radians to degrees
        index_B = 0
        print(index_B)
        score = 0

        for field in experiment_dict.keys():
            trp.B = field*gauss2mhz
            evalues = trp.eval(trp.D, trp.E, trp.B, trp.theta, trp.phi, mol_basis=False)
            transitions = evalues[2] - evalues[0], evalues[2] - evalues[1], evalues[1] - evalues[0]
            #print(transitions)

            for tr in transitions:
                if round(tr, 3) in experiment_dict[field]:
                    score += 0.3

        index_Theta += 1
        print('index_Theta ', index_Theta)

    index_Phi += 1
    print('index_Phi ', index_Phi)
    score = score / float(datapointsN)
    best_scores_list = scoring(score, best_scores_list)

    if score in best_scores_list:
        a,_ = np.where(best_scores_list == score)
        best_scores_angles[a, :] = (Phi_deg, Theta_deg, score)

print(best_scores_angles)
