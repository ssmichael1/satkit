# Quaternions

Quaternions are a mathematical concept that extends the idea of complex numbers to four dimensions.
They are used in many fields, including computer graphics, robotics, and physics. In the context of
this toolkit, they represent rotations in 3D space and are operationally equivalent to rotation matrices.

They have several advantages over rotation matrices, including:

- They are more compact, requiring only four numbers to represent a rotation instead of nine.
- They are more numerically stable, avoiding the gimbal lock problem.
- They are easier to interpolate between, making them useful for animations.

!!! warning
    Quaternions in this module perform a *right-handed* rotation of the vector. You will occasionally see (especially in attitude control texts) a quaternion convention that does a right-handed rotation of the coordinate system, which is a left-handed rotation of the vector.

For an excellent description of quaternions, see [here](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation).

::: satkit.quaternion
