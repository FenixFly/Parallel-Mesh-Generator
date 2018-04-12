#include "grainmesh.h"

grain::GrainMesh::GrainMesh()
{

}

grain::GrainMesh::~GrainMesh()
{

}

int grain::GrainMesh::getVerticesCount()
{
    return vertices.size();
}

std::vector<vec3d>* grain::GrainMesh::getVertices()
{
    return &vertices;
}

std::vector<char>* grain::GrainMesh::getVerticesLabels()
{
    return &verticesLabels;
}

int grain::GrainMesh::getTrianglesCount()
{
    return triangles.size();
}

std::vector<vec3i>* grain::GrainMesh::getTriangles()
{
    return &triangles;
}

std::vector<char>* grain::GrainMesh::getTrianglesLabels()
{
    return &trianglesLabels;
}

int grain::GrainMesh::getTetraCount()
{
    return tetra.size();
}

std::vector<vec4i>* grain::GrainMesh::getTetra()
{
    return &tetra;
}

std::vector<char>* grain::GrainMesh::getTetraLabels()
{
    return &tetraLabels;
}

void grain::GrainMesh::setVertices(std::vector<vec3d> &newVertices)
{
    vertices.clear();
    vertices = newVertices;
}

void grain::GrainMesh::setVerticesLabels(std::vector<char> &newLabels)
{
    verticesLabels.clear();
    verticesLabels = newLabels;
}

void grain::GrainMesh::setTriangles(std::vector<vec3i> &newTriangles)
{
    triangles.clear();
    triangles = newTriangles;
}

void grain::GrainMesh::setTrianglesLabels(std::vector<char> &newLabels)
{
    trianglesLabels.clear();
    trianglesLabels = newLabels;
}

void grain::GrainMesh::setTetra(std::vector<vec4i> &newTetra)
{
    tetra.clear();
    tetra = newTetra;
}

void grain::GrainMesh::setTetraLabels(std::vector<char> &newLabels)
{
    tetraLabels.clear();
    tetraLabels = newLabels;
}

void grain::GrainMesh::makeStubTetra()
{
	vertices.clear();
	vertices = std::vector<vec3d>(4);
	vertices.at(0).x = 0.1; vertices.at(0).y = 0.1; vertices.at(0).z = 0.1;
	vertices.at(1).x = 1.0; vertices.at(1).y = 0.0; vertices.at(1).z = 0.0;
	vertices.at(2).x = 0.0; vertices.at(2).y = 1.0; vertices.at(2).z = 0.0;
	vertices.at(3).x = 0.0; vertices.at(3).y = 0.0; vertices.at(3).z = 1.0;

	triangles.clear();
	triangles = std::vector<vec3i>(4);
	triangles.at(0).x = 0; triangles.at(0).y = 1; triangles.at(0).z = 3;
	triangles.at(1).x = 0; triangles.at(1).y = 2; triangles.at(1).z = 1;
	triangles.at(2).x = 0; triangles.at(2).y = 3; triangles.at(2).z = 2;
	triangles.at(3).x = 1; triangles.at(3).y = 2; triangles.at(3).z = 3;
}
