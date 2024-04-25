# Quaternions

**Quaternions** are an efficient way of representing 3-dimensional rotations. The **satkit** package includes its own implementation of quaternions.  They are the default output class when computing rotations between coordinate frames.

Quaternions are operationally equivalent to 3x3 rotation matrices, sometimes called **direction-cosine matrices** or **DCMs**. They have the advantage of being more computationally efficient, and unlike **DCMs** be easily renormalized such that they do not lose their *unitary nature* as multiplies are compounded. 

For an excellent overview of quaternions, see:<br/>
<https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation>

The quaternion class in this python package is a thin python wrapper around the unit quaternion module provided by the rust [nalgebra](https://docs.rs/nalgebra/latest/nalgebra/) package

## Quaternion Rotation

The **satkit** package represents a quaternion rotation by a left-sided multiply of a quaternion by a vector, similar to a left-sided multiply of a **DCM** by a vector.  Compounded rotations are represented by multiplications of quaternions. The right-most quaternion represents the first applied rotation, follwed by rotations represented by the quaternion on the immediate left, as with the **DCM**.

Let us define rotation matrices $R_x$, $R_y$, and $R_z$ that represent *right-handed* rotations of a vector about the $\hat{x}$, $\hat{y}$, and $\hat{z}$ unit vectors, respectively:


$$
R_x(\theta)~=~\left [ \begin{array}{ccc} 1 & 0 & 0 \\ 0 & \cos(\theta) & -\sin(\theta) \\ 0 & \sin(\theta) & \cos(\theta) \end{array} \right ]
$$

<br/>

$$
R_y(\theta)~=~\left [ \begin{array}{ccc} \cos(\theta) & 0 & \sin(\theta) \\ 0 & 1 & 0 \\ -\sin(\theta) & 0 & \cos(\theta) \end{array} \right ]
$$

<br/>

$$
R_z(\theta)~=~\left [ \begin{array}{ccc} \cos(\theta) & -\sin(\theta) & 0 \\ \sin(\theta) & \cos(\theta) & 0 \\ 0 & 0 & 1 \end{array} \right ]
$$

These equivalent rotations are defined in python as ```satkit.quaternion.rotx```, ```satkit.quaternion.roty```, and ```satkit.quaternion.rotz``` functions, respectively.  The functions take as an input the angle of rotation in radians.

The python code below computes the rotation of the $\hat{x}$ vector by $\pi/2$ about the $\hat{z}$ axis using both the traditional rotation matrices and then with the satkit quaternion.  In both cases, since this is a right-handed rotation of a vector, the resulting vector is $\hat{y}$.

```python
import satkit as sk
import math as m
import numpy as np

# Create the xhat vector
xhat = np.array([1, 0, 0], np.float64)

# Rotation matrix that rotates about zhat axis by input angle in radians
rotz = lambda x:np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0],[0, 0, 1]])

# Traditional rotation using matrix multiply
# Matrix multiply should produce yhat: [0, 1, 0]
yhat = rotz(m.pi/2) @ xhat

# Equivalent rotation using quaternions
yhat = sk.quaternion.rotz(m.pi/2) * xhat

# Extrat equivalent rotation from quaternion
# and do it via matrix multiplication
yhat = sk.quaternion.rotz(m.pi/2).to_rotation_matrix() @ xhat

```

## Compounding Rotations

Quaternions can be multiplied by one another, which is functionally equivalent to multiplying rotation matrices, and follows the same order of operations.

This is shown in the example code below:

```python
import satkit as sk
import math as m
import numpy as np

# Create the xhat vector
xhat = np.array([1, 0, 0], np.float64)

# Rotation matrix that rotates about xhat axis by input angle in radians
rotx = lambda x: np.array([[1, 0, 0],[0, np.cos(x), -np.sin(x)],[0,np.sin(x),np.cos(x)]])
# Rotation matrix that rotates about zhat axis by input angle in radians
rotz = lambda x: np.array([[np.cos(x), -np.sin(x), 0], [ np.sin(x), np.cos(x), 0],[0, 0, 1]])

# Rotating xhat about zhat axis by pi/2 then xhat axis by pi/2 produces zhat:
r1 = rotx(m.pi/2)
r2 = rotz(m.pi/2)
zhat_r = r1 @ r2 @ xhat

# Same thing with quaternions
q1 = sk.quaternion.rotx(m.pi/2)
q2 = sk.quaternion.rotz(m.pi/2)
zhat_q = q1 * q2 * xhat
```

