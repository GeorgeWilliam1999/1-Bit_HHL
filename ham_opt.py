import trackhhl.toy.simple_generator as toy
import trackhhl.hamiltonians.simple_hamiltonian as hamiltonian
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle
import numpy as np

N_MODULES = 5
#test bounds of LX and LY
LX = float("+inf")
LY = float("+inf")
Z_SPACING = 1.0

detector = toy.SimpleDetectorGeometry(
    module_id=list(range(N_MODULES)),
    lx=[LX]*N_MODULES,
    ly=[LY]*N_MODULES,
    z=[i+Z_SPACING for i in range(N_MODULES)]
)

generator = toy.SimpleGenerator(
    detector_geometry=detector,
    theta_max=np.pi/6
)

N_PARTICLES = 60
event = generator.generate_event(N_PARTICLES)

ham = hamiltonian.SimpleHamiltonian(
    epsilon=1e-3,
    gamma=2.0,
    delta=1.0
)

ham.construct_hamiltonian(event=event)
