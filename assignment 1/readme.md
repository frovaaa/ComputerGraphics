# Team Members
Frova Davide  
Jamila Oubenali

# Issues encountered

## Dot product must be strictly positive -> Specular (creating too much light at weird spots) AND diffuse (absorbing light)

We didn't think that we needed to check whether the dot product was negative at first, because we wrongly imagined that
every intersection point would not lead to a negative dot product.
As a result, some areas were too dark, and we also had some new "specular" highlights with odd colors. The explanation is that
because we are raising cos alpha to the k, if k is odd, it could add light intensity.

The issue was present for both specular and diffuse components. In the case of the diffuse component, the issue was that by accepting negative results, we would have negative light intensity, which is impossible; but in our case it was underflowing to a different color, instead of being clamped to 0 (black).

## Using degrees instead of radians

After seeing that our spheres would get deformed significantly, when put near the edges of the "camera plane", we realized that we were calling the C++ `tan` function with degrees instead of radians. After fixing this, the spheres deformation degree was significantly reduced.

## Confusing vectors and points

We misunderstood the "meaning" of a vector by thinking that its "position" in space mattered when really only its
magnitude and direction do. Because of this, we tried to compute our own "inverse" ray of direction (when calling the `PhongModel` function), but it was wrong
conceptually because a point isn't a vector, and it could've been wrong in practice because it would only work when the
camera was at the origin (0,0,0).

## Darker image than expected

If we compare the images produced by our code with the reference solution images, we can see that our images are darker than the reference images; even the various parameters for positions, color and lights are the same.
We found that by using `0.6` as the lights intensity instead of `0.4` we get a more similar image to the reference solution.
