import trackhhl.toy.simple_generator as toy
import trackhhl.hamiltonians.simple_hamiltonian as hamiltonian
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



N_MODULES = 5
#test bounds of LX and LY
LX = 3.0#float("+inf")
LY = 3.0#float("+inf")
Z_SPACING = 1.0

detector = toy.SimpleDetectorGeometry(
    module_id=list(range(N_MODULES)),
    lx=[LX]*N_MODULES,
    ly=[LY]*N_MODULES,
    z=[i+Z_SPACING for i in range(N_MODULES)]
)

generator = toy.SimpleGenerator(
    detector_geometry=detector,
    theta_max=np.pi/12
)

N_PARTICLES = 12
event = generator.generate_event(N_PARTICLES)

show_tracks = True; show_hits = True; show_modules = True; equal_axis = True; s_hits=10

fig = plt.figure()
fig.set_size_inches(12,8)
ax = plt.axes(projection='3d')

ax.set_facecolor('#CCCCCC')
if show_hits:
    hit_x, hit_y, hit_z = [], [], []
    for hit in event.hits:
        hit_x.append(hit.x)
        hit_y.append(hit.y)
        hit_z.append(hit.z)
    ax.scatter3D(hit_x, hit_y, hit_z,s=s_hits,c='navy')

if show_modules:
    for module in event.modules:
        p = Rectangle((-module.lx/2, -module.ly/2), module.lx, module.ly,alpha=.2,edgecolor='#20B2AA')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=module.z)

if show_tracks:
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    ts = []
    for track in event.tracks:
        pvx, pvy, pvz = track.mc_info.primary_vertex
        phi = track.mc_info.theta
        theta = track.mc_info.phi
        tx1 = max((x_lim[0] - pvx)/(np.sin(theta)*np.cos(phi)),0)
        tx2 = max((x_lim[1] - pvx)/(np.sin(theta)*np.cos(phi)),0)
        ty1 = max((y_lim[0] - pvy)/(np.sin(theta)*np.sin(phi)),0)
        ty2 = max((y_lim[1] - pvy)/(np.sin(theta)*np.sin(phi)),0)
        tz1 = max((z_lim[0] - pvz)/(np.cos(theta)),0)
        tz2 = max((z_lim[1] - pvz)/(np.cos(theta)),0)
        ts.append(min(max(tx1,tx2),max(ty1,ty2), max(tz1,tz2)))

    for track, t in zip(event.tracks, ts):
        pvx, pvy, pvz = track.mc_info.primary_vertex
        phi = track.mc_info.theta
        theta = track.mc_info.phi
        ax.plot((pvx,pvx + t*np.sin(theta)*np.cos(phi)),
        (pvy,pvy+ t*np.sin(theta)*np.sin(phi)),
        (pvz,z_lim[1]), alpha=0, color = '#BA82EB')

if equal_axis:
    set_axes_equal(ax)
    ax.set_box_aspect([10,6,6])
ax.set_proj_type('ortho')

ax.view_init(vertical_axis='y')
fig.set_tight_layout(True)
#ax.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', alpha=0.2) 
ax.axis('on')
ax.set_title(f"Generated event\n{len(event.modules)} modules\n{len(event.tracks)} tracks - {len(event.hits)} hits - {(len(event.tracks)**2)*(len(event.modules)-1) } Doublets")

#num_x_ticks = len(ax.get_xticks())
#num_y_ticks = len(ax.get_yticks())
#num_z_ticks = len(ax.get_zticks())

#ax.set_xticklabels([''] * num_x_ticks)
#ax.set_yticklabels([''] * num_y_ticks)
#ax.set_zticklabels([''] * num_z_ticks)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

track_lines = []

def animate(i):
    if i < len(event.tracks):

        track = event.tracks[i]
        pvx, pvy, pvz = track.mc_info.primary_vertex
        phi = track.mc_info.theta
        theta = track.mc_info.phi
        t = ts[i] 

        line, = ax.plot((pvx, pvx + t * np.sin(theta) * np.cos(phi)),
                           (pvy, pvy + t * np.sin(theta) * np.sin(phi)),
                           (pvz, z_lim[1]), alpha=1, color='#BA82EB')
        track_lines.append(line)

    return track_lines

# Replace the line calling plt.show() with:
ani = animation.FuncAnimation(fig, animate, frames=len(event.tracks), interval=400)
ani.save("animation.gif", writer="imagemagick")
plt.show()

fig = plt.figure()
fig.set_size_inches(12,8)
ax = plt.axes(projection='3d')

ax.set_facecolor('#CCCCCC')
if show_hits:
    hit_x, hit_y, hit_z = [], [], []
    for hit in event.hits:
        hit_x.append(hit.x)
        hit_y.append(hit.y)
        hit_z.append(hit.z)
    ax.scatter3D(hit_x, hit_y, hit_z,s=s_hits,c='navy')

if show_modules:
    for module in event.modules:
        p = Rectangle((-module.lx/2, -module.ly/2), module.lx, module.ly,alpha=.2,edgecolor='#20B2AA')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=module.z)

if show_tracks:
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    ts = []
    for track in event.tracks:
        pvx, pvy, pvz = track.mc_info.primary_vertex
        phi = track.mc_info.theta
        theta = track.mc_info.phi
        tx1 = max((x_lim[0] - pvx)/(np.sin(theta)*np.cos(phi)),0)
        tx2 = max((x_lim[1] - pvx)/(np.sin(theta)*np.cos(phi)),0)
        ty1 = max((y_lim[0] - pvy)/(np.sin(theta)*np.sin(phi)),0)
        ty2 = max((y_lim[1] - pvy)/(np.sin(theta)*np.sin(phi)),0)
        tz1 = max((z_lim[0] - pvz)/(np.cos(theta)),0)
        tz2 = max((z_lim[1] - pvz)/(np.cos(theta)),0)
        ts.append(min(max(tx1,tx2),max(ty1,ty2), max(tz1,tz2)))

    for track, t in zip(event.tracks, ts):
        pvx, pvy, pvz = track.mc_info.primary_vertex
        phi = track.mc_info.theta
        theta = track.mc_info.phi
        ax.plot((pvx,pvx + t*np.sin(theta)*np.cos(phi)),
        (pvy,pvy+ t*np.sin(theta)*np.sin(phi)),
        (pvz,z_lim[1]), alpha=0, color = '#BA82EB')

if equal_axis:
    set_axes_equal(ax)
    ax.set_box_aspect([10,6,6])
ax.set_proj_type('ortho')

ax.view_init(vertical_axis='y')
fig.set_tight_layout(True)
#ax.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', alpha=0.2) 
ax.axis('on')
ax.set_title(f"Generated event\n{len(event.modules)} modules\n{len(event.tracks)} tracks - {len(event.hits)} hits - {(len(event.tracks)**2)*(len(event.modules)-1) } Doublets")

#num_x_ticks = len(ax.get_xticks())
#num_y_ticks = len(ax.get_yticks())
#num_z_ticks = len(ax.get_zticks())

#ax.set_xticklabels([''] * num_x_ticks)
#ax.set_yticklabels([''] * num_y_ticks)
#ax.set_zticklabels([''] * num_z_ticks)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

track_lines = []

def animate(i):
    if i < len(event.tracks):

        track = event.tracks[i]
        pvx, pvy, pvz = track.mc_info.primary_vertex
        phi = track.mc_info.theta
        theta = track.mc_info.phi
        t = ts[i] 

        line, = ax.plot((pvx, pvx + t * np.sin(theta) * np.cos(phi)),
                           (pvy, pvy + t * np.sin(theta) * np.sin(phi)),
                           (pvz, z_lim[1]), alpha=0, color='#BA82EB')
        track_lines.append(line)

    return track_lines

# Replace the line calling plt.show() with:
ani = animation.FuncAnimation(fig, animate, frames=len(event.tracks), interval=400)
#ani.save("animation.gif", writer="imagemagick")
plt.show()