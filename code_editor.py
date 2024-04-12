import streamlit as st
from streamlit_ace import st_ace

"# Running `pyscript`"

"## Input"

value = """
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from pyscript import display

# First create the x and y coordinates of the points.
n_angles = 36
n_radii = 8
min_radius = 0.25
radii = np.linspace(min_radius, 0.95, n_radii)

angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += np.pi / n_angles

x = (radii * np.cos(angles)).flatten()
y = (radii * np.sin(angles)).flatten()
z = (np.cos(radii) * np.cos(3 * angles)).flatten()

# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(x, y)

# Mask off unwanted triangles.
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                            y[triang.triangles].mean(axis=1))
                < min_radius)

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang, z, shading='flat')
fig1.colorbar(tpc)
ax1.set_title('tripcolor of Delaunay triangulation, flat shading')

display(fig1, target="mpl")
"""

code = st_ace(
    value=value,
    language='python', 
    theme='tomorrow_night',
    tab_size= 4,
    font_size=16, height=200
)

"*****"
"## Output"

html = f"""
<html>
  <head>
    <link rel="stylesheet" href="https://pyscript.net/latest/pyscript.css" />
    <link rel="stylesheet" href="./custom.css" />
    <script defer src="https://pyscript.net/latest/pyscript.js"></script>
  </head>
  <body>
    <section class="pyscript">
        <div id="mpl"></div>
        <py-config>packages = [ "matplotlib" ]</py-config>
        <py-script terminal>{code}</py-script>
    </section>
  </body>
</html>
"""

st.components.v1.html(html, width=700, height=600, scrolling=True)