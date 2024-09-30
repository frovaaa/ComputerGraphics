# How to build

1. Compile `main.cpp`

Run `g++ main.cpp`. This will create a `.out` file

2. Run `.out` file

Run `./a.out`. This will generate a `result.ppm` file which you can then visualize

# TODOs left
- [] check what happens when we put the camera inside the sphere
# Issues encountered
## dot product must be strictly positive -> Specular (creating too much light at weird spots) AND diffuse (absorbing light)
We didn't think that we needed to check whether the dot product was negative at first, because we wrongly imagined that 
every intersection point would not lead to a negative dot product.
As a result, some areas were too dark, and we also had some new "specular" highlights with odd colors. The explanation is that
because we are raising cos alpha to the k, if k is odd, it could add light intensity.


## Using degrees instead of radians
Initially, we used degrees without giving it much thought. After seeing that our spheres would get squished
when we moved it to the edges of the space, we reconsidered our code and thought to check if the function %TODO insert name%
we used took.


## Confusing vectors and points, 
We misunderstood the "meaning" of a vector by thinking that its "position" in space mattered when really only its 
magnitude and direction do. Because of this, we tried to compute our own "inverse" ray of direction, but it was wrong 
conceptually because a point isn't a vector, and it could've been wrong in practice because it would only work when the
camera was at the origin (0,0,0).   


## do we have to change values for light intensity? it's not enough? (cf bottom of green ball for example)


