/**
@file main.cpp

~ Frova Davide
~ Jamila Oubenali

*/

#include <algorithm>
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

class Box;

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
    if (equalFloats(0, NdotD, 0.0000001)) {
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

// Assignment 3: Triangle class
class Triangle : public Object {
 private:
  Plane *plane;
  glm::vec3 a;  // p1
  glm::vec3 b;  // p2
  glm::vec3 c;  // p3
  glm::vec3 normal;

 public:
  Triangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, Material material)
      : a(a), b(b), c(c) {
    this->normal = glm::cross((b - a), (c - a));
    this->plane = new Plane(a, this->normal);
    this->material = material;
  }

  Triangle(glm::vec3 a, glm::vec3 b, glm::vec3 c) : a(a), b(b), c(c) {
    this->normal = glm::cross((b - a), (c - a));
    this->plane = new Plane(a, this->normal);
  }

  glm::vec3 getA() { return this->a; }

  glm::vec3 getB() { return this->b; }

  glm::vec3 getC() { return this->c; }

  // Bonus assignment
  glm::vec3 getCentroid() { return (a + b + c) / 3.0f; }

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

      // TODO: Fix this
      if ((lambda1 >= 0 && lambda2 >= 0 && lambda3 >= 0) &&
          (lambda1 + lambda2 + lambda3) <= 1.0 + 1e-6) {
        hit.normal = glm::normalize(this->normal);
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

/* Bonus assignment: Class representing a bounding box, used in the kdtree */
class Box {
 private:
  // box_min contains x.min, y.min, z.min
  glm::vec3 box_min;
  glm::vec3 box_max;

 public:
  Box(glm::vec3 box_min, glm::vec3 box_max)
      : box_min(box_min), box_max(box_max) {}

  /**
   * The function checks if we intersect the bounding box or not. If we did,
   * then we run the intersect with the mesh contained inside of the box.
   * @param ray
   * @return Hit
   */
  bool intersect(Ray ray) {
    /* Compute the t_values:
     * TODO: Correct this definition comment
     * t_min_i is the point on the t ray which first intersects the bounding
     * box's side parallel to the i axis
     */
    // TODO: Check if this is the most optimized way to do it
    // We will always keep the biggest t_min value and the smallest t_max values
    float t_min = -INFINITY;
    float t_max = INFINITY;

    float t_1;
    float t_2;

    float entry;
    float exit;
    // Iterate on all axis
    for (int i = 0; i < 3; ++i) {
      if (ray.direction[i] != 0) {
        t_1 = ((box_min[i] - ray.origin[i]) / ray.direction[i]);
        t_2 = ((box_max[i] - ray.origin[i]) / ray.direction[i]);
        entry = std::min(t_1, t_2);
        exit = std::max(t_1, t_2);

        // TODO: Check efficiency ? wrt to if statement or ternary
        t_min = std::max(entry, t_min);
        t_max = std::min(exit, t_max);

      } else {
        if (ray.origin[i] < box_min[i] || ray.origin[i] > box_max[i]) {
          return false;
        }
      }
    }
    if (t_min > t_max || t_max < 0) {
      return false;
    }
    // If the bounding box was hit, run the intersect function on the mesh
    return true;
  }
};

class KDTreeNode {
 public:
  // Axis along which the node splits the space
  // x -> 0, y -> 1, z -> 2
  // if it is a leaf node, axis is -1
  int axis;

  // Splitting position
  float split;

  // Bounding box of the node
  Box *boundingBox;

  // Left and right children
  KDTreeNode *left;
  KDTreeNode *right;

  // List of triangles in the leaf node
  std::vector<Triangle *> triangles;

  // TODO: Fix this, we can use a union to save memory but right now it gives
  // errors
  // We can use a union to save memory as we will only use one of the two
  // fields union {
  //   // Internal node
  //   struct {
  //     KDTreeNode *left;
  //     KDTreeNode *right;
  //   };
  //   // Leaf node
  //   // TODO: Could be a pointer to nextTriangle instead of a vector
  //   std::vector<Triangle *> triangles;
  // };

  // Constructor for internal node
  KDTreeNode(int axis, float split, KDTreeNode *left, KDTreeNode *right,
             Box *boundingBox)
      : axis(axis),
        split(split),
        left(left),
        right(right),
        boundingBox(boundingBox) {}

  // Constructor for leaf node
  KDTreeNode(std::vector<Triangle *> triangles, Box *boundingBox)
      : axis(-1), split(0), triangles(triangles), boundingBox(boundingBox) {}

  // Destructor
  ~KDTreeNode() {
    // If it is a leaf node, we delete the triangles
    if (axis == -1) {
      // We clear the vector of triangles
      triangles.clear();
    } else {
      // If it is an internal node, we delete the left and right children
      delete left;
      delete right;
    }
  }

  // Function to check if the node is a leaf node
  bool isLeaf() { return axis == -1; }

  // Function to check if a ray intersects the bounding box of the node
  bool intersectsBoundingBox(Ray ray) { return boundingBox->intersect(ray); }
};

// Function to compute the bounding Box of a given vector of triangles
// using the minimum and maximum points of the triangles
Box *computeBoundingBox(std::vector<Triangle *> triangles) {
  glm::vec3 box_min = glm::vec3(INFINITY);
  glm::vec3 box_max = glm::vec3(-INFINITY);

  for (auto &triangle : triangles) {
    box_min = glm::min(box_min,
                       glm::min(triangle->getA(),
                                glm::min(triangle->getB(), triangle->getC())));
    box_max = glm::max(box_max,
                       glm::max(triangle->getA(),
                                glm::max(triangle->getB(), triangle->getC())));
  }

  return new Box(box_min, box_max);
}

int MAX_TRIANGLES_PER_LEAF = 5;
KDTreeNode *buildKDTree(std::vector<Triangle *> triangles, int depth = 0) {
  // If there are no triangles, we return a null pointer
  if (triangles.empty()) {
    return nullptr;
  }

  // If there are less or equal triangles than the maximum number of triangles
  // per leaf, we create a leaf node
  if (triangles.size() <= MAX_TRIANGLES_PER_LEAF) {
    // First we compute the bounding box of the leaf node
    Box *boundingBox = computeBoundingBox(triangles);
    // Then we create the leaf node
    return new KDTreeNode(triangles, boundingBox);
  } else {
    // We have an internal node

    // First we find along which axis we will split the space
    // We just alternate between the three axis on each depth level
    int axis = depth % 3;

    // Now we need to choose the splitting position
    // We can use similar techniques as in the BVH described in the slides
    // We will use the median of centers of mass of the triangles along the axis
    // as our splitting position

    /*
      To do this we sort the triangles based on their centroids
      std::sort takes start and end addresses of the vector
      and a comparator function (can be a lambda function)
      The lambda function is defined as folows:
      [] contains the capture list, which contains the outside variables that
      are available inside the lambda function
      then we have the arguments of the lambda function, and finally the body
    */
    std::sort(triangles.begin(), triangles.end(),
              [axis](Triangle *a, Triangle *b) {
                return a->getCentroid()[axis] < b->getCentroid()[axis];
              });

    // Now we find the median and split the triangles into two halves
    size_t median = triangles.size() / 2;
    std::vector<Triangle *> left(triangles.begin(), triangles.begin() + median);
    std::vector<Triangle *> right(triangles.begin() + median, triangles.end());

    // We create the bounding box of the internal node
    Box *boundingBox = computeBoundingBox(triangles);

    // Recursively build the left and right children
    KDTreeNode *leftChild = buildKDTree(left, depth + 1);
    KDTreeNode *rightChild = buildKDTree(right, depth + 1);

    // We create the internal node
    return new KDTreeNode(axis, triangles[median]->getCentroid()[axis],
                          leftChild, rightChild, boundingBox);
  }
}

// Function to check if a ray intersects the KDTree
Hit intersectKDTree(KDTreeNode *node, Ray ray) {
  Hit hit;
  hit.hit = false;
  hit.intersection = glm::vec3(0);
  hit.distance = INFINITY;
  hit.normal = glm::vec3(0);
  hit.object = nullptr;

  // If the node is nullptr, we do not hit
  if (node == nullptr) {
    return hit;
  }

  // Check if the ray intersects with the node bounding box
  // if not, we do not hit
  if (!node->intersectsBoundingBox(ray)) {
    return hit;
  }

  // The ray intersects the bounding box
  // So now we check if it is a leaf node or an internal node

  // If the node is a leaf, we check intersections with the triangles with the
  // classic closest intersection strategy
  // TODO: Could be optimized by using BVH for the triangles
  if (node->isLeaf()) {
    for (auto &triangle : node->triangles) {
      Hit triangleHit = triangle->intersect(ray);
      // If the current hit is false, or the triangleHit distance is smaller
      // we set the hit to the triangleHit
      if (triangleHit.hit &&
          (!hit.hit || triangleHit.distance < hit.distance)) {
        hit = triangleHit;
      }
    }
    // We then return the hit
    return hit;
  } else {
    // If we are an internal node, we need to check recursively intersection for
    // the left and right children and return the closest intersection (if any)

    Hit leftHit = intersectKDTree(node->left, ray);
    Hit rightHit = intersectKDTree(node->right, ray);

    // Check leftHit
    if (leftHit.hit && (leftHit.distance < hit.distance)) {
      hit = leftHit;
    }
    // Check rightHit
    if (rightHit.hit && (rightHit.distance < hit.distance)) {
      hit = rightHit;
    }
    // Finally we return the hit
    return hit;
  }
}

class Mesh : public Object {
 private:
  // List of triangles
  std::vector<Triangle *> triangles;
  // File path to the obj file
  std::string objPath;
  // Bonus assignment: KDTree
  KDTreeNode *kdTreeRoot;

 public:
  Mesh(std::string objPath, Material material) : objPath(objPath) {
    this->material = material;
    this->loadObj();
    // this->kdTreeRoot = buildKDTree(this->triangles);
  }

  Mesh(std::string objPath) : objPath(objPath) {
    this->loadObj();
    // this->kdTreeRoot = buildKDTree(this->triangles);
  }

  Mesh(std::string objPath, glm::mat4 transformationMatrix, Material material)
      : objPath(objPath) {
    this->material = material;
    this->transformationMatrix = transformationMatrix;
    this->loadObj();
    // this->kdTreeRoot = buildKDTree(this->triangles);
  }

  Mesh(std::string objPath, glm::mat4 transformationMatrix) : objPath(objPath) {
    this->transformationMatrix = transformationMatrix;
    this->loadObj();
    // this->kdTreeRoot = buildKDTree(this->triangles);
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
    std::vector<glm::vec3> faces;

    std::string line;
    // Read the file line by line
    while (std::getline(file, line)) {
      // Craete a string stream from the line
      std::stringstream ss(line);
      std::string type;
      // Read the first word of the line
      ss >> type;

      // Check the type of the line
      // Could be a vertex, a normal, face or the smooth shading option
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
        */
        glm::vec3 face;
        std::string vertex;
        for (int i = 0; i < 3; i++) {
          ss >> vertex;
          std::stringstream vss(vertex);
          std::string index;
          std::getline(vss, index, '/');
          face[i] = std::stoi(index);
        }
        faces.push_back(face);
      }
    }

    // Create the triangles from the vertices and faces
    for (int i = 0; i < faces.size(); i++) {
      glm::vec3 a = vertices[faces[i].x - 1];
      glm::vec3 b = vertices[faces[i].y - 1];
      glm::vec3 c = vertices[faces[i].z - 1];

      Triangle *triangle = new Triangle(a, b, c, this->material);
      triangle->setTransformation(this->transformationMatrix);

      this->triangles.push_back(triangle);
    }

    // In the end we close the file
    file.close();

    std::cout << "Number of triangles: " << this->triangles.size() << std::endl;
    std::cout << "Number of vertices: " << vertices.size() << std::endl;
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
 Function that returns the color of the closest object intersected by the
 given Ray Checks if the ray intersects with any of the objects in the scene
 If so, return the color of the cloest object that got hit, if not returns
 the black color (0.0, 0.0, 0.0)
 @param ray Ray that should be traced through the scene
 @return Color at the intersection point
 */
glm::vec3 trace_ray(Ray ray) {
  Hit closest_hit;

  closest_hit.hit = false;
  closest_hit.distance = INFINITY;

  // For each object in the scene, we run the intersect function
  // If the hit is positive, we check if the distance is the smallest seen
  // so far. This will give us the closes_hit from the camera Maybe we will
  // need to check for negative values as they would result smaller than
  // positive ones
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
// scan all triangles of each mesh and keep the extreme values to compute
// bounding box
/* Notes:
 * - function to create bounding box
 * - when tracing ray, check if it hits bounding box of object. If yes, then
 * try to intersect with the object (s) in that box. Otherwise, skip to next
 * bounding box
 * */
/**
 Function defining the scene
 */
void sceneDefinition() {
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

  /* Add walls */
  // Left wall
  objects.push_back(new Plane(back_left_p, x_norm));
  // Bottom wall
  objects.push_back(new Plane(back_left_p, y_norm));
  // Right wall
  objects.push_back(new Plane(top_right_p, x_norm));
  // Front wall
  objects.push_back(new Plane(top_right_p, z_norm));

  /* Add meshes */
  glm::mat4 armadilloTrans = glm::translate(glm::vec3(-4.0f, -3.0f, 10.0f));
  glm::mat4 armadilloRot =
      glm::rotate(glm::mat4(1.0f), (float)0.0, glm::vec3(1.0f, 0.0f, 0.0f));
  glm::mat4 armadilloScale = glm::scale(glm::vec3(1.0f, 1.0f, 1.0f));
  glm::mat4 armadilloTraMat = armadilloTrans * armadilloRot * armadilloScale;
  // Mesh *armadillo = new Mesh("meshes/armadillo.obj", armadilloTraMat);
  // armadillo->addMeshToScene();
  // objects.push_back(armadillo);

  glm::mat4 bunnyTrans = glm::translate(glm::vec3(0.0f, -3.0f, 8.0f));
  glm::mat4 bunnyRot = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f),
                                   glm::vec3(1.0f, 0.0f, 0.0f));
  glm::mat4 bunnyScale = glm::scale(glm::vec3(1.0f, 1.0f, 1.0f));
  glm::mat4 bunnyTraMat = bunnyTrans * bunnyRot * bunnyScale;
  Mesh *bunny = new Mesh("meshes/bunny_small.obj", bunnyTraMat);
  bunny->addMeshToScene();
  objects.push_back(bunny);

  glm::mat4 lucyTrans = glm::translate(glm::vec3(4.0f, -3.0f, 10.0f));
  glm::mat4 lucyRot = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f),
                                  glm::vec3(1.0f, 0.0f, 0.0f));
  glm::mat4 lucyScale = glm::scale(glm::vec3(1.0f, 1.0f, 1.0f));
  glm::mat4 lucyTraMat = lucyTrans * lucyRot * lucyScale;
  // Mesh *lucy = new Mesh("meshes/lucy.obj", lucyTraMat);
  // lucy->addMeshToScene();
  // objects.push_back(lucy);
  cout << "Number of objects: " << objects.size() << endl;
}

glm::vec3 toneMapping(glm::vec3 intensity) {
  /*  ---- Exercise 3-----
   Implement a tonemapping strategy and gamma correction for a correct
   display.
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
  int width = 2048;   // width of the image
  int height = 1536;  // height of the image
  float fov = 90;     // field of view

  // variable for defining the scene (building kdtrees ecc)
  clock_t t1 = clock();
  sceneDefinition();  // Let's define a scene
  t1 = clock() - t1;
  cout << "It took " << ((float)t1) / CLOCKS_PER_SEC
       << " seconds to define the scene." << endl;

  Image image(width,
              height);  // Create an image where we will store the result

  // Size of Pixel which depends on width and fov
  float S = (2 * tan(glm::radians(fov / 2))) / width;

  // How much to translate from the 3D origin center of the plane to get to
  // the point at i,j
  float X = -S * width / 2;
  float Y = S * height / 2;

  clock_t t = clock();  // variable for keeping the time of the rendering
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
      image.setPixel(i, j,
                     glm::clamp(toneMapping(trace_ray(ray)), glm::vec3(0.0),
                                glm::vec3(1.0)));

      // TODO: Remove when testing performance
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
