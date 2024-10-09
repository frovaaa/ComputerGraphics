/**
@file main.cpp

~ Frova Davide
~ Jamila Oubenali

*/

#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "Image.h"
#include "Material.h"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

using namespace std;

/**
 Class representing a single ray.
 */
class Ray {
 public:
  glm::vec3 origin;     ///< Origin of the ray
  glm::vec3 direction;  ///< Direction of the ray
  /**
   Contructor of the ray
   @param origin Origin of the ray
   @param direction Direction of the ray
   */
  Ray(glm::vec3 origin, glm::vec3 direction)
      : origin(origin), direction(direction) {}
};

class Object;

/**
 Structure representing the event of hitting an object
 */
struct Hit {
  bool hit;          ///< Boolean indicating whether there was or there was no
                     ///< intersection with an object
  glm::vec3 normal;  ///< Normal vector of the intersected object at the
                     ///< intersection point
  glm::vec3 intersection;  ///< Point of Intersection
  float distance;  ///< Distance from the origin of the ray to the intersection
                   ///< point
  Object *object;  ///< A pointer to the intersected object
};

/**
 General class for the object
 */
class Object {
 protected:
  glm::mat4
      transformationMatrix;  ///< Matrix representing the transformation from
                             ///< the local to the global coordinate system
  glm::mat4 inverseTransformationMatrix;  ///< Matrix representing the
                                          ///< transformation from the global to
                                          ///< the local coordinate system
  glm::mat4 normalMatrix;  ///< Matrix for transforming normal vectors from the
                           ///< local to the global coordinate system

 public:
  glm::vec3 color;    ///< Color of the object
  Material material;  ///< Structure describing the material of the object
  /** A function computing an intersection, which returns the structure Hit */
  virtual Hit intersect(Ray ray) = 0;

  /** Function that returns the material struct of the object*/
  Material getMaterial() { return material; }

  /** Function that set the material
   @param material A structure describing the material of the object
  */
  void setMaterial(Material material) { this->material = material; }
  /** Functions for setting up all the transformation matrices
  @param matrix The matrix representing the transformation of the object in the
  global coordinates */
  void setTransformation(glm::mat4 matrix) {
    transformationMatrix = matrix;

    /* ----- Exercise 2 ---------
    Set the two remaining matrices

    inverseTransformationMatrix =
    normalMatrix =

    */
  }
};

/**
 Implementation of the class Object for sphere shape.
 */
class Sphere : public Object {
 private:
  float radius;      ///< Radius of the sphere
  glm::vec3 center;  ///< Center of the sphere

 public:
  /**
   The constructor of the sphere
   @param radius Radius of the sphere
   @param center Center of the sphere
   @param color Color of the sphere
   */
  Sphere(float radius, glm::vec3 center, glm::vec3 color)
      : radius(radius), center(center) {
    this->color = color;
  }

  Sphere(float radius, glm::vec3 center, Material material)
      : radius(radius), center(center) {
    this->material = material;
  }

  /** Implementation of the intersection function*/
  Hit intersect(Ray ray) {
    Hit hit;
    hit.hit = false;
    hit.intersection = glm::vec3(0);
    hit.distance = 0;
    hit.normal = glm::vec3(0);
    hit.object = this;

    // If the origin of the primary ray (camera) is not at the world 0,0,0
    // We translate the center of the sphere to match the offest and act like if
    // the camera is at 0,0,0
    glm::vec3 c = this->center - ray.origin;

    // If the origin of the ray is inside the Sphere, we return black color
    // In this case the vector c is equal to the translated new center of the
    // sphere C
    if (glm::length(c) <= this->radius) {
      // The origin of the ray is inside the Sphere
      return hit;
    }

    float a = glm::dot(c, ray.direction);

    float D = sqrt(pow(glm::length(c), 2) - pow(a, 2));

    // Epsilon used for equality check between floats
    const float EPSILON = 0.01;

    // Cases
    if (D < this->radius) {
      // Two solutions
      float b = sqrt(pow(this->radius, 2) - pow(D, 2));
      float t1 = a + b;
      float t2 = a - b;

      // Now we need to check if the t are > 0, otherwise the sphere is behind
      // the camera and it doesn't need to be rendered
      if (t1 > 0 || t2 > 0) {
        // At least one of the two intersections is in front of the camera
        // I set at INFINITY if the t is < 0
        t1 = t1 > 0 ? t1 : INFINITY;
        t2 = t2 > 0 ? t2 : INFINITY;

        // We choose only the intersection that is closest to the origin of the
        // ray
        float t = t1 < t2 ? t1 : t2;

        hit.intersection = ray.direction * t;
        hit.distance = t;
        hit.hit = true;
      }
      // Float comparison: if the radius is equal
    } else if (std::abs(D - this->radius) < EPSILON) {
      // t = a+b	In this case b == 0 so a == t
      if (a > 0) {
        // One solution
        hit.intersection = ray.direction * a;

        hit.distance = a;
        hit.hit = true;
      }
    }
    // If the ray hit the object, compute the normal
    if (hit.hit) {
      // the coordinates are already shifted here
      hit.normal = glm::normalize(hit.intersection - c);
    }
    return hit;
  }
};

class Plane : public Object {
 private:
  glm::vec3 normal;
  glm::vec3 point;

 public:
  Plane(glm::vec3 point, glm::vec3 normal) : point(point), normal(normal) {}
  Plane(glm::vec3 point, glm::vec3 normal, Material material)
      : point(point), normal(normal) {
    this->material = material;
  }
  Hit intersect(Ray ray) {
    Hit hit;
    hit.hit = false;

    /*



     Excercise 1 - Plane-ray intersection




     */

    return hit;
  }
};

class Cone : public Object {
 private:
  Plane *plane;

 public:
  Cone(Material material) {
    this->material = material;
    plane = new Plane(glm::vec3(0, 1, 0), glm::vec3(0.0, 1, 0));
  }
  Hit intersect(Ray ray) {
    Hit hit;
    hit.hit = false;

    /*  ---- Exercise 2 -----

     Implement the ray-cone intersection. Before intersecting the ray with the
     cone, make sure that you transform the ray into the local coordinate
     system. Remember about normalizing all the directions after
     transformations.

    */

    /* If the intersection is found, you have to set all the critical fields in
    the Hit strucutre Remember that the final information about intersection
    point, normal vector and distance have to be given in the global coordinate
    system.

    hit.hit = true;
    hit.object = this;
    hit.intersection =
    hit.normal =
    hit.distance =

     */

    return hit;
  }
};

/**
 Light class
 */
class Light {
 public:
  glm::vec3 position;  ///< Position of the light source
  glm::vec3 color;     ///< Color/intentisty of the light source
  Light(glm::vec3 position) : position(position) { color = glm::vec3(1.0); }

  Light(glm::vec3 position, glm::vec3 color)
      : position(position), color(color) {}
};

vector<Light *> lights;  ///< A list of lights in the scene
glm::vec3 ambient_light(0.1, 0.1, 0.1);
vector<Object *> objects;  ///< A list of all objects in the scene

/** Function for computing color of an object according to the Phong Model
 @param point A point belonging to the object for which the color is computed
 @param normal A normal vector the the point
 @param view_direction A normalized direction from the point to the
 viewer/camera
 @param material A material structure representing the material of the object
*/
glm::vec3 PhongModel(glm::vec3 point, glm::vec3 normal,
                     glm::vec3 view_direction, Material material) {
  // Illumination intensity
  glm::vec3 color(0.0);

  // Iterate over all light sources
  for (int i = 0; i < lights.size(); ++i) {
    // light direction
    glm::vec3 light_direction = glm::normalize(lights[i]->position - point);

    // Angle between the normal and the light direction
    // No need to check negative as we clamp the value
    float phi = glm::clamp(glm::dot(normal, light_direction), 0.0f, 1.0f);
    glm::vec3 reflected_direction = ((2.0f * normal) * phi) - light_direction;

    float RdotV =
        glm::clamp(glm::dot(reflected_direction, view_direction), 0.0f, 1.0f);

    // Diffuse
    glm::vec3 diffuse_color = material.diffuse;
    glm::vec3 diffuse = diffuse_color * glm::vec3(phi);

    // Specular illumination
    glm::vec3 specular =
        material.specular * glm::vec3(pow(RdotV, material.shininess));

    /*  ---- Exercise 3-----

     Include light attenuation due to the distance to the light source.

    */

    // Add the contribution of the light source to the final color
    color += lights[i]->color * (diffuse + specular);
  }
  // Add ambient illumination
  color += material.ambient * ambient_light;
  // The final color has to be clamped so the values do not go beyond 0 and 1.
  color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0));
  return color;
}

/**
 Function that returns the color of the closest object intersected by the given
 Ray Checks if the ray intersects with any of the objects in the scene If so,
 return the color of the cloest object that got hit, if not returns the black
 color (0.0, 0.0, 0.0)
 @param ray Ray that should be traced through the scene
 @return Color at the intersection point
 */
glm::vec3 trace_ray(Ray ray) {
  Hit closest_hit;

  closest_hit.hit = false;
  closest_hit.distance = INFINITY;

  // For each object in the scene, we run the intersect function
  // If the hit is positive, we check if the distance is the smallest seen so
  // far. This will give us the closes_hit from the camera Maybe we will need to
  // check for negative values as they would result smaller than positive ones
  for (int k = 0; k < objects.size(); k++) {
    Hit hit = objects[k]->intersect(ray);
    if (hit.hit == true && hit.distance < closest_hit.distance)
      closest_hit = hit;
  }

  glm::vec3 color(0.0);

  // If the ray hit something, save the color of the object hit
  // and compute the color using the Phong model
  // Otherwise, return black color
  if (closest_hit.hit) {
    color = PhongModel(closest_hit.intersection, closest_hit.normal,
                       glm::normalize(-ray.direction),
                       closest_hit.object->getMaterial());
  } else {
    color = glm::vec3(0.0, 0.0, 0.0);
  }
  return color;
}

/**
 Function defining the scene
 */
void sceneDefinition() {
  Material red_specular;
  red_specular.diffuse = glm::vec3(1.0f, 0.3f, 0.3f);
  red_specular.ambient = glm::vec3(0.01f, 0.03f, 0.03f);
  red_specular.specular = glm::vec3(0.5);
  red_specular.shininess = 10.0f;
  objects.push_back(new Sphere(0.5, glm::vec3(-1, -2.5, 6), red_specular));

  // Definition of the blue sphere
  Material blue_dark;
  blue_dark.diffuse = glm::vec3(0.7f, 0.7f, 1.0f);
  blue_dark.ambient = glm::vec3(0.07f, 0.07f, 0.1f);
  blue_dark.specular = glm::vec3(0.6f);
  blue_dark.shininess = 100.0f;
  objects.push_back(new Sphere(1.0f, glm::vec3(1, -2, 8), blue_dark));

  // Definition of the green sphere
  Material green;
  green.diffuse = glm::vec3(0.7f, 0.9f, 0.7f);
  green.ambient = glm::vec3(0.07f, 0.09f, 0.07f);
  green.specular = glm::vec3(0.0f);
  green.shininess = 0.0f;
  objects.push_back(new Sphere(1.0f, glm::vec3(2, -2, 6), green));

  // Define lights
  lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(0.4)));
  lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.4)));
  lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4)));
}

glm::vec3 toneMapping(glm::vec3 intensity) {
  /*  ---- Exercise 3-----

   Implement a tonemapping strategy and gamma correction for a correct display.

  */

  return intensity;
}

int main(int argc, const char *argv[]) {
  clock_t t = clock();  // variable for keeping the time of the rendering

  int width = 1024;  // width of the image
  int height = 768;  // height of the image
  float fov = 90;    // field of view

  sceneDefinition();  // Let's define a scene

  Image image(width, height);  // Create an image where we will store the result

  // Size of Pixel which depends on width and fov
  float S = (2 * tan(glm::radians(fov / 2))) / width;

  // How much to translate from the 3D origin center of the plane to get to the
  // point at i,j
  float X = -S * width / 2;
  float Y = S * height / 2;

  for (int i = 0; i < width; i++)
    for (int j = 0; j < height; j++) {
      float dx = X + i * S + S / 2;
      float dy = Y - j * S - S / 2;
      float dz = 1;

      // Definition of the ray
      glm::vec3 origin(0, 0, 0);
      glm::vec3 direction(dx, dy, dz);
      direction = glm::normalize(direction);

      Ray ray(origin, direction);  // ray traversal

      image.setPixel(i, j, trace_ray(ray));

      /*  ---- Exercise 3-----
      After implementing the tonemapping function
      use the following line

      image.setPixel(i, j,
              glm::clamp(toneMapping(trace_ray(ray)), glm::vec3(0.0),
                        glm::vec3(1.0)));
      */
    }

  t = clock() - t;
  cout << "It took " << ((float)t) / CLOCKS_PER_SEC
       << " seconds to render the image." << endl;
  cout << "I could render at " << (float)CLOCKS_PER_SEC / ((float)t)
       << " frames per second." << endl;

  // Writing the final results of the rendering
  if (argc == 2) {
    image.writeImage(argv[1]);
  } else {
    image.writeImage("./result.ppm");
  }

  return 0;
}
