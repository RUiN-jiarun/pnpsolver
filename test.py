import numpy as np
from mypnp import scale_adaptive_pnp

points2D = np.loadtxt('example/C/point2Ds.txt').transpose().astype('float32').copy()
points3D = np.loadtxt('example/C/point3Ds.txt').transpose().astype('float32').copy()
priors = np.loadtxt('example/C/priors.txt').reshape(1, -1).astype('float32').copy()

camera = {"model_name": "PINHOLE", "params": [1727.69, 1727.69, 388, 521]}

Tow = np.array([[-1.14216520e+00,  4.29204776e+00,  1.23306060e+01, -1.61736221e+02],
             [-1.30553074e+01, -2.88119428e-01, -1.10894186e+00, -1.05676075e+00],
            [-9.24277981e-02, -1.23787525e+01,  4.30071730e+00,  1.33246706e+00],
             [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]).astype('float32').copy()

center = np.array([0.0, 0.0, 0.0]).reshape(1, -1).astype('float32').copy()

print(points2D.shape)
print(points3D.shape)
print(priors.shape)
print(Tow.shape)
print(center.shape)

pnp_option = {
    'error_thres': 12,
    'inlier_ratio': 0.01,
    'confidence': 0.9999,
    'max_iter': 10000,
    'local_optimal': True,
    'fix_x': True,
    'fix_y': False,
    'fix_z': False
}

ret = scale_adaptive_pnp(points2D, points3D, priors, Tow, center, camera, pnp_option)

print(ret["qvec"])
print(ret["tvec"])
print(ret["scale"])