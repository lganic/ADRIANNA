This test will demonstrate the efficacy of training values in a neural network live, as opposed to pre-training. This will be done using a simplified version of the reward training pipeline laid out, since we are just developing a simple case here. 

We will be using a simple physics object in a box. The physics object can apply a force to move itself left, right, up, and down. If the physics object hits the box it is contained in, it gets hurt. Inside the box is a rotating target. The physics object will get a reward corresponding to how close it is to the moving target. The inputs to the network will be such:

* Top distance raycast
* Bottom distance raycast
* Left distance raycast
* Right distance raycast
* Delta X to target
* Delta Y to target
* X Velocity
* Y Velocity

Then our outputs will be

* Top Force
* Bottom Force
* Side Force
* Down Force

These will be mapped to the output by a linear matrix with no bias. The initialization of the matrix will be zero (at least for now), and then we will look at how the system behaves.

To make things a little more interesting, the forces will be clamped to always be greater than or equal to zero