import numpy as np
from trackhhl.toy import simple_generator as toy
from trackhhl.hamiltonians import simple_hamiltonian as hamiltonian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
If we adjust the vertex more than 0.0001 we have major loss in accuracy 
So from this information we need to know the radial transfrom frame of reference to an accuracy of 0.0001
Would need about 3000 scans over this space to have an accurate result
'''

def cartesian_to_spherical_vectorized(cartesian_coords, translation=(0, 0, 0)):
    x, y, z = cartesian_coords[:, 0] - translation[0], cartesian_coords[:, 1] - translation[1], cartesian_coords[:, 2] - translation[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return r, theta, phi

accuracy_test = []
resolution_scale = []
pv = []

for acc in range(100):

    N_MODULES = 4
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

    N_PARTICLES = 50
    N_EVENTS = 10
    SIGMA = (0.5,0,0)
    events = generator.generate_event(N_PARTICLES, N_EVENTS, SIGMA)

    event = events[0]
    event_hits = []
    events_store = []
    for event_ in events:
        for hit in event_.hits:
            event_hits.append([hit.x,hit.y,hit.z,hit.track_id])
        events_store.append(event_hits)
    structured_array = np.array(events_store)
    cartesian_coords = structured_array[:, :, :3]  

    hist = []
    theta_axis = []
    phi_axis = []
    bins = (500, 500)
    for i, vert in enumerate(generator.primary_vertices):
        r, theta, phi = cartesian_to_spherical_vectorized(cartesian_coords[i], vert)
        hist.append(np.histogram2d(theta, phi, bins=bins)[0])
        theta_axis.append(np.histogram2d(theta, phi, bins=bins)[1])
        phi_axis.append(np.histogram2d(theta, phi, bins=bins)[2])

    data_with_histogram_info = []
    for i, histogram in enumerate(hist):
        x_indices, y_indices = np.where(histogram >= 4)
        histogram_indices = np.full((len(x_indices),3), generator.primary_vertices[i])
        data_points = np.column_stack((theta_axis[i][x_indices], phi_axis[i][y_indices], histogram_indices))
        data_with_histogram_info.append(data_points)
    final_data = np.vstack(data_with_histogram_info)
    print(1 - abs(len(final_data)-(N_PARTICLES*N_EVENTS))/(N_PARTICLES*N_EVENTS))
    accuracy_test.append(1 - abs(len(final_data)-(N_PARTICLES*N_EVENTS))/(N_PARTICLES*N_EVENTS))
    resolution_scale.append(abs(np.mean(generator.primary_vertices)/0.0001))
    pv.append(np.mean(generator.primary_vertices))


print(f"Method Accuracy for N_PARTICLES={N_PARTICLES} and N_EVENTS={N_EVENTS} is", np.mean(accuracy_test),"With STD:",np.std(accuracy_test), "And PV proxmity ratio needed is", np.mean(resolution_scale))
print("Mean PV", np.max(pv), np.min(pv))