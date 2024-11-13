/**
@file main.cpp

~ Frova Davide
~ Jamila Oubenali

*/

#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "Image.h"
#include "Material.h"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

using namespace std;

bool equalFloats(float a, float b, float EPSILON) {
  return (std::fabs(a - b) < EPSILON);
}

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

bool intersects_any_object(Ray, glm::vec3);

glm::vec3 trace_ray(Ray);

class Object;

/**
 Structure representing the event of hitting an object
 */
struct Hit {
  bool hit;  ///< Boolean indicating whether there was or there was no
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
    */
    /* Assignment 2: set matrices */
    inverseTransformationMatrix = glm::inverse(transformationMatrix);
    normalMatrix = glm::transpose(inverseTransformationMatrix);
  }
};

vector<Object *> objects;  ///< A list of all objects in the scene

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
    glm::vec3 c = center - ray.origin;

    float cdotc = glm::dot(c, c);
    float cdotd = glm::dot(c, ray.direction);

    Hit hit;

    float D = 0;
    if (cdotc > cdotd * cdotd) {
      D = sqrt(cdotc - cdotd * cdotd);
    }
    if (D <= radius) {
      hit.hit = true;
      float t1 = cdotd - sqrt(radius * radius - D * D);
      float t2 = cdotd + sqrt(radius * radius - D * D);

      float t = t1;
      if (t < 0) t = t2;
      if (t < 0) {
        hit.hit = false;
        return hit;
      }

      hit.intersection = ray.origin + t * ray.direction;
      hit.normal = glm::normalize(hit.intersection - center);
      hit.distance = glm::distance(ray.origin, hit.intersection);
      hit.object = this;
    } else {
      hit.hit = false;
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
    /* Assignment 2: intersect function for plane and ray*/
    Hit hit;
    hit.hit = false;
    hit.intersection = glm::vec3(0);
    hit.distance = 0;
    hit.normal = glm::vec3(0);
    hit.object = this;

    float NdotD = glm::dot(this->normal, ray.direction);
    // If the dotproduct is 0, the ray direction is parallel to the plane so
    // there is no intersection
    if (equalFloats(0, NdotD, 0.001)) {
      return hit;
    } else {
      float NdotP = glm::dot(this->normal, this->point);
      float NdotO = glm::dot(this->normal, ray.origin);

      float t = (NdotP - NdotO) / NdotD;

      if (t <= 0) {
        return hit;
      }
      hit.hit = true;
      hit.intersection = ray.origin + ray.direction * t;
      hit.distance = t;

      /* Compute whether the sign or the normal must be flipped or not
       based on the position of the camera, so that the plane's color is not
       "one-view" only but actually renders on both sides. */
      hit.normal = glm::dot(-(ray.direction), this->normal) < 0
                       ? -(this->normal)
                       : this->normal;
      return hit;
    }
  }
};

class Cone : public Object {
 private:
  Plane *plane;

 public:
  Cone(Material material) {
    this->material = material;
    // The point must be in the center of the cone's base
    plane = new Plane(glm::vec3(0, 1, 0), glm::vec3(0.0, 1, 0));
  }

  Hit intersect(Ray ray) {
    /*  ---- Exercise 2 -----
     * Implement the ray-cone intersection. Before intersecting the ray with the
     * cone, make sure that you transform the ray into the local coordinate
     * system. Remember about normalizing all the directions after
     * transformations.
     */

    /* Assignment 2: Intersect function for cone and ray */

    // Radius of the cone's base
    float BASE_RADIUS = 1.0f;
    Hit hit;
    hit.hit = false;
    hit.intersection = glm::vec3(0);
    hit.distance = 0;
    hit.normal = glm::vec3(0);
    hit.object = this;

    /* Conversion to homogenous and local coordinates */
    // Conversion to Homogeneous coordinates
    glm::vec4 localRayOrigin = glm::vec4(ray.origin, 1.0f);
    glm::vec4 localRayDirection = glm::vec4(ray.direction, 0.0f);

    // Apply the inverse matrix to bring the ray to local coordinates
    localRayOrigin = this->inverseTransformationMatrix * localRayOrigin;
    localRayDirection =
        glm::normalize(this->inverseTransformationMatrix * localRayDirection);

    // We need a local ray to pass to the plane intersect function
    Ray *localRay =
        new Ray(glm::vec3(localRayOrigin), glm::vec3(localRayDirection));

    // Checking if the ray hits the plane (base of the cone)
    Hit hitPlane = this->plane->intersect(*localRay);
    // Center of the cone's base
    glm::vec3 baseCenter = glm::vec3(0.0f, 1.0f, 0.0f);

    // Check if the base of the cone was hit
    if (hitPlane.hit) {
      /* If the cone is not hit at all, the fields other than 'hit' are never
       * used, so we can assign them like this now. Reminder: fields of the
       * Hit structure are in global coordinates
       */
      hit.intersection = glm::vec3(this->transformationMatrix *
                                   glm::vec4(hitPlane.intersection, 1.0f));
      hit.normal = glm::normalize(
          glm::vec3(this->normalMatrix * glm::vec4(hitPlane.normal, 0.0f)));
      hit.distance = glm::distance(hit.intersection, ray.origin);
      hit.hit = hitPlane.hit;

      /* If the radius is superior to one, it is not part of the cone's base */
      // hitPlane.intersection is in local coordinates
      if (hit.hit &&
          glm::distance(hitPlane.intersection, baseCenter) > BASE_RADIUS) {
        hit.hit = false;
      }
    }
    /* Equation of the cone is x^2 + z^2 - y^2 = 0
     * Search for gamma(t) = x^2 + z^2 - y^2*/

    /* Solve with the quadratic formula */
    // Quadratic formula coefficients
    float a = pow(localRayDirection.x, 2) + pow(localRayDirection.z, 2) -
              pow(localRayDirection.y, 2);
    float b = 2 * (localRayDirection.x * localRayOrigin.x +
                   localRayDirection.z * localRayOrigin.z -
                   localRayDirection.y * localRayOrigin.y);
    float c = pow(localRayOrigin.x, 2) + pow(localRayOrigin.z, 2) -
              pow(localRayOrigin.y, 2);

    // b^2 - 4ac
    float delta = pow(b, 2) - 4 * (a * c);
    // If delta < 0, no solution
    if (delta < 0) {
      return hit;
    }

    float t = INFINITY;
    // Compute both t
    float t_1 = (-b + sqrt(delta)) / (2 * a);
    float t_2 = (-b - sqrt(delta)) / (2 * a);
    // Choose the smallest one
    if (t_1 > 0 && t_1 < t) t = t_1;
    if (t_2 > 0 && t_2 < t) t = t_2;
    if (t == INFINITY) {
      return hit;
    }

    // Possible point of intersection on the side of the cone
    glm::vec4 candidate = localRayOrigin + (localRayDirection * t);
    // Setting homogenous coordinate for candidate point
    candidate[3] = 1.0f;

    // Check that the possible intersection point on the side of the cone is
    // legal
    if (candidate.y > 1 || candidate.y < 0) {
      return hit;
    }

    float distanceSide = glm::distance(
        glm::vec3(this->transformationMatrix * candidate), ray.origin);
    // If the plane/base wasn't hit, set distance to infinity
    float distanceBase = hit.hit ? hit.distance : INFINITY;

    // If the cone's base was hit, check what part of the cone was hit first
    if (distanceBase < distanceSide) {
      return hit;
    }
    // Otherwise, the side was hit first
    // Transform coordinates to global
    hit.intersection = glm::vec3(this->transformationMatrix * candidate);
    hit.distance = glm::distance(hit.intersection, ray.origin);

    // Compute the normal in the local coordinates using the gradient
    glm::vec3 localNormal = glm::normalize(
        glm::vec3(2 * candidate.x, -2 * candidate.y, 2 * candidate.z));

    // Transforming the normal into global coordinates
    hit.normal = glm::normalize(
        glm::vec3(this->normalMatrix * glm::vec4(localNormal, 0.0f)));

    /* Our first guess to compute the normal using a 180° rotation around the
    axis (swapping the direction only at the end).
     * It isn't completely correct (cf shadow of laid down cone)
    glm::mat4 rot_n = glm::rotate(glm::mat4(1.0f), (float)glm::radians(180.0f),
    glm::vec3(0.0f, 1.0f, 0.0f)); hit.normal = glm::normalize(rot_n *
    (glm::normalize(this->inverseTransformationMatrix *
    glm::vec4(hit.intersection, 1.0f)))); hit.normal =
    -glm::normalize(glm::vec3(glm::vec4(hit.normal, 0.0f) *
    this->normalMatrix));
    */
    hit.hit = true;
    return hit;
  }
};

// Assignment 3: Triangle class
class Triangle : public Object {
 private:
  Plane *plane;
  glm::vec3 a;  // p1
  glm::vec3 b;  // p2
  glm::vec3 c;  // p3
  glm::vec3 normal;
  std::vector<glm::vec3> smoothNormals;  // vector of normals for smooth shading

 public:
  Triangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, Material material)
      : a(a), b(b), c(c) {
    this->normal = glm::cross((b - a), (c - a));
    this->plane = new Plane(a, this->normal);
    this->material = material;
  }

  Triangle(glm::vec3 a, glm::vec3 b, glm::vec3 c,
           std::vector<glm::vec3> smoothNormals)
      : a(a), b(b), c(c), smoothNormals(smoothNormals) {
    this->normal = glm::cross((b - a), (c - a));
    this->plane = new Plane(a, this->normal);
  }

  Triangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, Material material,
           std::vector<glm::vec3> smoothNormals)
      : a(a), b(b), c(c), smoothNormals(smoothNormals) {
    this->normal = glm::cross((b - a), (c - a));
    this->plane = new Plane(a, this->normal);
    this->material = material;
  }

  Triangle(glm::vec3 a, glm::vec3 b, glm::vec3 c) : a(a), b(b), c(c) {
    this->normal = glm::cross((b - a), (c - a));
    this->plane = new Plane(a, this->normal);
  }

  /* override the setTransformation method so that we can update
   * the a, b, c verteces, the normal and recompute the plane
   */
  void setTransformation(glm::mat4 matrix) {
    // Set the transformation matrices for the triangle
    this->transformationMatrix = matrix;
    this->inverseTransformationMatrix =
        glm::inverse(this->transformationMatrix);
    this->normalMatrix = glm::transpose(this->inverseTransformationMatrix);

    // Transform all the verteces and re-compute the normal and the plane
    this->a = glm::vec3(this->transformationMatrix * glm::vec4(this->a, 1.0f));
    this->b = glm::vec3(this->transformationMatrix * glm::vec4(this->b, 1.0f));
    this->c = glm::vec3(this->transformationMatrix * glm::vec4(this->c, 1.0f));
    this->normal = glm::cross((this->b - this->a), (this->c - this->a));
    this->plane = new Plane(this->a, this->normal);

    // Transform all the smooth normals if they are present
    for (int i = 0; i < this->smoothNormals.size(); i++) {
      this->smoothNormals[i] = glm::vec3(
          this->normalMatrix * glm::vec4(this->smoothNormals[i], 0.0f));
    }
  }

  Hit intersect(Ray ray) {
    Hit hit;
    hit.hit = false;
    hit.intersection = glm::vec3(0);
    hit.distance = 0;
    hit.normal = this->normal;
    hit.object = this;

    // Checking if the ray hits the plane
    Hit hitPlane = this->plane->intersect(ray);

    if (hitPlane.hit) {
      /* We hit the plane, so now we check the barycentric coordinates
       * To see if we are inside the triangle
       */
      hit.intersection = hitPlane.intersection;
      hit.distance = hitPlane.distance;
      hit.hit = hitPlane.hit;

      glm::vec3 n1 = glm::cross((this->b - hit.intersection),
                                (this->c - hit.intersection));
      float dot1 = glm::dot(this->normal, n1);
      float lambda1 = dot1 / glm::pow(glm::length(this->normal), 2);

      glm::vec3 n2 = glm::cross((this->c - hit.intersection),
                                (this->a - hit.intersection));
      float dot2 = glm::dot(this->normal, n2);
      float lambda2 = dot2 / glm::pow(glm::length(this->normal), 2);

      glm::vec3 n3 = glm::cross((this->a - hit.intersection),
                                (this->b - hit.intersection));
      float dot3 = glm::dot(this->normal, n3);
      float lambda3 = dot3 / glm::pow(glm::length(this->normal), 2);

      if ((lambda1 >= 0 && lambda2 >= 0 && lambda3 >= 0) &&
          (lambda1 + lambda2 + lambda3) <= 1.0 + 1e-6) {
        if (this->smoothNormals.size() > 0) {
          hit.normal = glm::normalize(lambda1 * this->smoothNormals[0] +
                                      lambda2 * this->smoothNormals[1] +
                                      lambda3 * this->smoothNormals[2]);
        } else {
          hit.normal = glm::normalize(this->normal);
        }
        return hit;
      } else {
        hit.hit = false;
      }
    }
    return hit;
  }
};

struct Face {
  std::vector<int> vertices;
  std::vector<int> normals;
};

class Mesh : public Object {
 private:
  // List of triangles
  std::vector<Triangle *> triangles;
  // File path to the obj file
  std::string objPath;

  bool smoothShading = false;

 public:
  Mesh(std::string objPath, Material material) : objPath(objPath) {
    this->material = material;
    this->loadObj();
  }

  Mesh(std::string objPath) : objPath(objPath) { this->loadObj(); }

  Mesh(std::string objPath, glm::mat4 transformationMatrix, Material material)
      : objPath(objPath) {
    this->material = material;
    this->transformationMatrix = transformationMatrix;
    this->loadObj();
  }

  Mesh(std::string objPath, glm::mat4 transformationMatrix) : objPath(objPath) {
    this->transformationMatrix = transformationMatrix;
    this->loadObj();
  }

  vector<Triangle *> getTriangles() { return this->triangles; }

  /*
    Function to load the obj file of the and create the triangles
  */
  void loadObj() {
    std::ifstream file(objPath);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << objPath << std::endl;
      return;
    }

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<Face> faces;

    std::string line;
    // Read the file line by line
    while (std::getline(file, line)) {
      // Create a string stream from the line
      std::stringstream ss(line);
      std::string type;
      // Read the first word of the line
      ss >> type;

      // Check the type of the line
      if (type == "v") {
        glm::vec3 vertex;
        ss >> vertex.x >> vertex.y >> vertex.z;
        vertices.push_back(vertex);
      } else if (type == "vn") {
        glm::vec3 normal;
        ss >> normal.x >> normal.y >> normal.z;
        normals.push_back(normal);
      } else if (type == "f") {
        /*
          Face composed by v/vt/vn
          (vertex index, texture, normal index)
          vt could be empty (like in our case)
          Read the three vertices of the face
          and store them in the faces vector
          The index of the vertices is 1-based so we need to subtract 1

          The face line could also be just f v v v
          So we need to check if the line is in the first format
        */
        Face face;
        std::string vertex;
        // Read space separated vertices
        while (ss >> vertex) {
          // Creates a string stream from the vertex string
          std::stringstream vss(vertex);
          std::string index;
          // Reads the vertex index before the first '/'
          // If there is no '/' it means that the line is in the second format
          // So we just read the entire vertex
          std::getline(vss, index, '/');
          face.vertices.push_back(std::stoi(index) - 1);
          // Check if the line is in the first format
          // We do this by checking if the next character is a '/'
          if (vss.peek() == '/') {
            // If yes, we ignore the first '/' as we ignore the texture index
            vss.ignore();
            // Then, we read the normal index and add it to the list
            std::getline(vss, index, '/');
            face.normals.push_back(std::stoi(index) - 1);
          }
        }
        faces.push_back(face);
      } else if (type == "s") {
        // Smooth shading option
        std::string option;
        ss >> option;
        this->smoothShading = (option == "1");
      }
    }

    // Create the triangles from the vertices and faces
    for (int i = 0; i < faces.size(); i++) {
      glm::vec3 a = vertices[faces[i].vertices[0]];
      glm::vec3 b = vertices[faces[i].vertices[1]];
      glm::vec3 c = vertices[faces[i].vertices[2]];

      Triangle *triangle;
      if (smoothShading && faces[i].normals.size() == 3) {
        std::vector<glm::vec3> smoothNormals;
        for (int j = 0; j < 3; j++) {
          smoothNormals.push_back(normals[faces[i].normals[j]]);
        }
        triangle = new Triangle(a, b, c, this->material, smoothNormals);
      } else {
        triangle = new Triangle(a, b, c, this->material);
      }
      triangle->setTransformation(this->transformationMatrix);
      this->triangles.push_back(triangle);
    }

    file.close();

    std::cout << "Number of triangles: " << this->triangles.size() << std::endl;
    std::cout << "Number of vertices: " << vertices.size() << std::endl;
    std::cout << "Number of normals: " << normals.size() << std::endl;
  }

  Hit intersect(Ray ray) {
    Hit hit;
    hit.hit = false;
    hit.intersection = glm::vec3(0);
    hit.distance = 0;
    hit.normal = glm::vec3(0);
    hit.object = this;

    for (int i = 0; i < this->triangles.size(); i++) {
      Hit triangleHit = this->triangles[i]->intersect(ray);
      if (triangleHit.hit &&
          (!hit.hit || triangleHit.distance < hit.distance)) {
        hit = triangleHit;
      }
    }
    return hit;
  }

  void addMeshToScene() {
    for (int i = 0; i < this->triangles.size(); i++) {
      objects.push_back(this->triangles[i]);
    }
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
glm::vec3 ambient_light(0.02f, 0.02f, 0.02f);

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

    // Assignment 4: Check if intersects to create shades
    Ray ray_shade(point + (light_direction * 0.001f), light_direction);

    // If there is any object between the intersection point and the light, we
    // do not contribute the color
    if (intersects_any_object(ray_shade, lights[i]->position)) {
      continue;
    }

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

    /* Assignment 2: Distance attenuation*/
    float att_d = 1;

    float r = glm::distance(point, lights[i]->position);
    if (r > 0) {
      float alpha1 = 0.01;
      float alpha2 = 0.01;
      float alpha3 = 0.01;
      att_d = 1 / (alpha1 + alpha2 * r + alpha3 * r * r);
    }
    // Add the contribution of the light source to the final color
    color += lights[i]->color * (diffuse + specular) * att_d;
  }

  // Add ambient illumination
  color += material.ambient * ambient_light;
  // The final color has to be clamped so the values do not go beyond 0 and 1.
  color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0));
  return color;
}
// Assignment 4
/**
 * Checks for intersection with any object
 * @param ray
 * @return true if there was an intersection, false otherwise
 */
bool intersects_any_object(Ray ray, glm::vec3 limit_point) {
  // For each object in the scene, we run the intersect function
  // If the hit is positive, we return true
  for (int i = 0; i < objects.size(); i++) {
    Hit hit = objects[i]->intersect(ray);
    if (hit.hit == true &&
        (hit.distance < glm::distance(ray.origin, limit_point))) {
      return true;
    }
  }
  return false;
}

/**
 Function that returns the color of the closest object intersected by the given
 Ray Checks if the ray intersects with any of the objects in the scene If so,
 return the color of the cloest object that got hit, if not returns the black
 color (0.0, 0.0, 0.0)
 @param ray Ray that should be traced through the scene
 @return Color at the intersection point
 */
glm::vec3 trace_ray(Ray ray, int current_depth) {
  if (current_depth >= 5) {
    return glm::vec3(0.0f, 0.0f, 0.0f);
  }

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

  // If the ray hit something, save the color of the object hit
  // and compute the color using the Phong model
  // Otherwise, return black color
  glm::vec3 color(0.0f);

  // Initialize the indices so we can use them in the final color computation
  // even if we don't refract
  float fresnel_refraction = 0.0f;
  float fresnel_reflection = 0.0f;
  // Base color values won't contribute to the end color
  glm::vec3 refracted_color = glm::vec3(0.0f);
  glm::vec3 reflected_color = glm::vec3(0.0f);
  /* If we hit, go further */
  if (closest_hit.hit) {
    // Compute reflection direction
    glm::vec3 reflection_direction = glm::normalize(glm::reflect(
        glm::normalize(ray.direction), glm::normalize(closest_hit.normal)));
    // Define reflection ray
    Ray reflected_ray(closest_hit.intersection + reflection_direction * 0.001f,
                      reflection_direction);
    // Compute the reflection color
    reflected_color = closest_hit.object->material.reflection *
                      trace_ray(reflected_ray, current_depth + 1);

    float fresnel_reflection = closest_hit.object->getMaterial().reflection;

    /* If the current object refracts the light */
    if (closest_hit.object->material.refracts_light) {
      float delta1 = 1.0f;
      float delta2 = closest_hit.object->getMaterial().refraction_index;
      // Angle between normal and ray.direction (i) == theta1
      bool from_outside = (glm::dot(closest_hit.normal, -ray.direction) > 0);
      glm::vec3 normal =
          from_outside ? closest_hit.normal : -closest_hit.normal;
      float cos_theta1 = glm::dot(normal, -ray.direction);

      /* If the ray is coming from inside the object, we invert delta 1 and
       * delta 2 to keep the formula coherent */
      if (!from_outside) {
        float temp = delta1;
        delta1 = delta2;
        delta2 = temp;
      }
      // Compute the refraction direction
      glm::vec3 refraction_direction =
          glm::refract(ray.direction, normal, delta1 / delta2);
      // Compute the refraction ray
      Ray refraction_ray(
          closest_hit.intersection + refraction_direction * 0.001f,
          refraction_direction);
      // Compute the color of the refraction ray
      refracted_color = trace_ray(refraction_ray, current_depth + 1);
      // Compute cos(theta2): We must invert the normal
      float cos_theta2 = glm::dot(-normal, refraction_direction);

      // Fresnel Effect
      // First part of the formula
      float comp1 = (delta1 * cos_theta1);
      float comp2 = (delta2 * cos_theta2);

      float second_comp1 = (delta1 * cos_theta2);
      float second_comp2 = (delta2 * cos_theta1);

      float first_power = pow(((comp1 - comp2) / (comp1 + comp2)), 2);

      float second_power = pow(
          ((second_comp1 - second_comp2) / (second_comp1 + second_comp2)), 2);

      float fr = 0.5f * (first_power + second_power);

      fresnel_reflection = fr;
      fresnel_refraction = 1.0 - fresnel_reflection;
    }

    /* If there was no reflection and/or refraction, it will not contribute to
     * the sum */
    color = PhongModel(closest_hit.intersection, closest_hit.normal,
                       glm::normalize(-ray.direction),
                       closest_hit.object->getMaterial()) +
            glm::clamp(fresnel_reflection, 0.0f, 1.0f) * reflected_color +
            glm::clamp(fresnel_refraction, 0.0f, 1.0f) * refracted_color;
  }
  return color;
}

/**
 Function defining the scene
 */
void sceneDefinition() {
  Material red_specular;
  red_specular.diffuse = glm::vec3(0.9f, 0.1f, 0.1f);
  red_specular.ambient = glm::vec3(0.1f, 0.03f, 0.03f);
  red_specular.specular = glm::vec3(0.5f);
  red_specular.shininess = 10.0f;

  Material blue_dark;
  blue_dark.diffuse = glm::vec3(0.1f, 0.1f, 0.8f);
  blue_dark.ambient = glm::vec3(0.01f, 0.01f, 0.9f);
  blue_dark.specular = glm::vec3(0.6f);
  blue_dark.shininess = 100.0f;

  Material green;
  green.diffuse = glm::vec3(0.2f, 0.9f, 0.2f);
  green.ambient = glm::vec3(0.01f, 0.3f, 0.01f);
  green.specular = glm::vec3(0.0f);
  green.shininess = 0.0f;

  Material white;
  white.diffuse = glm::vec3(0.9f);
  white.ambient = glm::vec3(0.1f);
  white.specular = glm::vec3(0.0f);
  white.shininess = 0.0f;

  /* Assignment 2: Yellow material for the highly specular cone*/
  Material yellow;
  yellow.diffuse = glm::vec3(0.2f, 0.2f, 0.0f);
  yellow.ambient = glm::vec3(0.003f, 0.003f, 0.0f);
  yellow.specular = glm::vec3(1.0);
  yellow.shininess = 100.0f;

  /* Assignment 4: mirror material */
  Material reflective;
  reflective.diffuse = glm::vec3(0.0f);
  reflective.ambient = glm::vec3(0.0f);
  reflective.specular = glm::vec3(0.0f);
  reflective.shininess = 0.5f;
  reflective.reflection = 1.0f;

  Material refractive;
  refractive.diffuse = glm::vec3(0.0f);
  refractive.ambient = glm::vec3(0.0f);
  refractive.specular = glm::vec3(0.0f);
  refractive.refracts_light = true;
  refractive.refraction_index = 2.0f;

  Material refractive_reflective;
  refractive_reflective.diffuse = glm::vec3(0.0f);
  refractive_reflective.ambient = glm::vec3(0.0f);
  refractive_reflective.specular = glm::vec3(0.0f);
  refractive_reflective.shininess = 0.5f;
  refractive_reflective.refracts_light = true;
  refractive_reflective.refraction_index = 2.0f;
  refractive_reflective.reflection = 1.0f;

  /* Add spheres */
  //    objects.push_back(new Sphere(2.5f, glm::vec3(-4, -0.5, 10), green));

  // Assignment 4: Refractive sphere
  objects.push_back(
      new Sphere(2.0f, glm::vec3(-3, -1, 8), refractive_reflective));

  objects.push_back(new Sphere(0.5, glm::vec3(-1, -2.5, 6), red_specular));
  objects.push_back(new Sphere(1.0f, glm::vec3(1, -2, 8), reflective));

  /* Define lights */
  lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(1.0f)));
  lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.1f)));
  lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4f)));

  /* Assignment 2: Planes */
  // Points at extremities of the box (top right and back left)
  glm::vec3 top_right_p = glm::vec3(15, 27, 30);
  glm::vec3 back_left_p = glm::vec3(-15, -3, -0.01);

  // Planes normals
  glm::vec3 x_norm = glm::vec3(1, 0, 0);
  glm::vec3 y_norm = glm::vec3(0, 1, 0);
  glm::vec3 z_norm = glm::vec3(0, 0, 1);

  // Left wall
  objects.push_back(new Plane(back_left_p, x_norm, red_specular));
  // Bottom wall
  objects.push_back(new Plane(back_left_p, y_norm, white));
  // Back wall
  // objects.push_back(new Plane(back_left_p, z_norm, blue_dark));

  // Right wall
  objects.push_back(new Plane(top_right_p, x_norm, blue_dark));
  // Above/top wall
  objects.push_back(new Plane(top_right_p, y_norm, red_specular));
  // Front wall
  objects.push_back(new Plane(top_right_p, z_norm, green));

  /* Assignment 2: Adding cones */

  /* Transformation matrices for the yellow cone */
  glm::mat4 translation_yellow_cone = glm::translate(glm::vec3(5, 9, 14));
  glm::mat4 rotation_yellow_cone =
      glm::rotate(glm::mat4(1.0f), (float)glm::radians(180.0f),
                  glm::vec3(0.0f, 0.0f, 1.0f));
  glm::mat4 scale_yellow_cone = glm::scale(glm::vec3(3.0f, 12.0f, 3.0f));
  glm::mat4 yellowConeTraMat =
      translation_yellow_cone * rotation_yellow_cone * scale_yellow_cone;

  // Define yellow cone
  Cone *yellow_cone = new Cone(yellow);
  // Set the transformation matrix for the yellow cone
  yellow_cone->setTransformation(yellowConeTraMat);

  /* Transformation matrices for the green cone */
  glm::mat4 translation_green_cone = glm::translate(glm::vec3(6, -3, 7));
  /* To compute the right angle to lay down the cone on the ground, we should
   * take arctan(b/|a|), here it's arctan(3) */
  glm::mat4 rotation_green_cone = glm::rotate(
      glm::mat4(1.0f), (float)glm::atan(3), glm::vec3(0.0f, 0.0f, 1.0f));
  glm::mat4 scale_green_cone = glm::scale(glm::vec3(1.0f, 3.0f, 1.0f));
  glm::mat4 greenConeTraMat =
      translation_green_cone * rotation_green_cone * scale_green_cone;

  Cone *green_cone = new Cone(green);
  green_cone->setTransformation(greenConeTraMat);

  /* Push the cones */
  objects.push_back(yellow_cone);
  objects.push_back(green_cone);

  cout << "Number of objects: " << objects.size() << endl;
}

glm::vec3 toneMapping(glm::vec3 intensity) {
  /*  ---- Exercise 3-----
   Implement a tonemapping strategy and gamma correction for a correct display.
  */
  /* Assignment 2: Tonemapping with power function and gamma correction */
  // Alpha has no constraints
  float alpha = 2.5f;
  // Beta must be less than 1
  float beta = 0.9f;
  intensity = glm::vec3(alpha * pow(intensity[0], beta),
                        alpha * pow(intensity[1], beta),
                        alpha * pow(intensity[2], beta));
  /* Gamma correction */
  // Usually gamma ~= 2.2, 1.8 for macs
  float gamma_inv = 1.0 / 2.2f;
  intensity =
      glm::vec3(pow(intensity[0], gamma_inv), pow(intensity[1], gamma_inv),
                pow(intensity[2], gamma_inv));

  // Other way of writing : intensity = glm::vec3(glm::pow(intensity,
  // glm::vec3(gamma_inv)));

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

      // image.setPixel(i, j, trace_ray(ray));
      image.setPixel(i, j,
                     glm::clamp(toneMapping(trace_ray(ray, 0)), glm::vec3(0.0),
                                glm::vec3(1.0)));

      // Print the progress of the rendering
      if (j % 10000 == 0) {
        float percentage = (float)(i * height + j) / (width * height) * 100;
        cout << "Progress: " << percentage << "%" << endl;
      }
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
