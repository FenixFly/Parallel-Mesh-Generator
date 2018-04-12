#pragma once
// Header for often includes //
#include <memory>
#include <vector>
//#include <iosfwd>

// TypeDefs //
typedef int Point4i[4];

struct vec3d
{   float x,y,z;   };
struct vec3i
{   int x,y,z;  };
struct vec4i
{   int x,y,z,w;    };

int version();
