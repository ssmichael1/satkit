(quaternion-api)=
# Quaternions

## Introduction

Quaternions are a mathematical concept that extends the idea of complex numbers to four dimensions.
They are used in many fields, including computer graphics, robotics, and physics.  In the context of 
this toolkit, they represent rotations in 3D space and are operationally equivalent to rotation matrices.

They have several advantages over rotation matrices, including:
- They are more compact, requiring only four numbers to represent a rotation instead of nine.
- They are more numerically stable, avoiding the gimbal lock problem.
- They are easier to interpolate between, making them useful for animations.

This toolkit provides a Class for working with quaternions, including methods for
creating, manipulating, and converting them.

```{warning}
Quaternions in this module perform a *right-handed* rotation of the $vector$.  You will occasionally see (especially in attitude control texts) a quaternion convention that does a right-handed rotation of the coordinate system, which is a left-handed rotation of the vector.  This abomination serves only to add confusion and must be struck down wherever it is seen.
```

For an excellent description of quaternions, see [here](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)

## Class Reference

```{eval-rst}
.. autoapiclass:: satkit.quaternion
   :members:
   :special-members:
```