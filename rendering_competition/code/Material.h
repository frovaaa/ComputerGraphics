//
//  Material.h
//  Raytracer
//
//  Created by Piotr Didyk on 14.07.21.
//

#ifndef Material_h
#define Material_h

#include "glm/glm.hpp"

/**
 Structure describing a material of an object
 */
struct Material {
  glm::vec3 ambient = glm::vec3(0.0);
  glm::vec3 diffuse = glm::vec3(1.0);
  glm::vec3 specular = glm::vec3(0.0);
  float shininess = 0.0;
  // Assignment 4
  float reflection = 0.0f;
  bool refracts_light = false;
  float refraction_index = 0.0f;

  // Bonus rendering competition
  // Apply or not the perlin noise to the material during phong model
  // calculation
  bool apply_perlin_noise = false;
  // Choose how many axis to use for the perlin noise
  // 2: use vec2
  // 3: use vec3
  int perlin_noise_type = 0;
  float perlin_noise_intensity = 0.3f;
  float perlin_noise_scale = 1.0f;
  bool perlin_noise_gradient = false;

  float perlin_noise_normal_intensity = 0.3f;
  float perlin_noise_normal_scale = 1.0f;
};

#endif /* Material_h */
