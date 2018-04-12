#pragma once
#include <iostream>
#include <set>
#include "helper_math.cuh"
#include "mymesh.h"
#include "mystl.h"
#include "meshcut.h"

class MeshSmooth
{
public:
	double edgelen;
	std::vector<std::set<int>*>* adjenctionList = nullptr;
	void smoothMesh(MyMesh * mesh, MySTL * stl);
private:
	void fillAdjunctionsList(MyMesh * mesh, std::vector<std::set<int>*> * &adjList);
	float3 findIntersectionWithSTL(float3 point, MySTL * stl);
	float3 getIntersectionPoint(float3 point, float3 p1, float3 p2, float3 p3);
	float hasIntersection(float3 point, float3 dst, float3 p1, float3 p2, float3 p3);

	void deletePoints(MyMesh * mesh, std::vector<int>* deletePoints);


	std::vector<int2>* generateOldNewPairs(MyMesh* mesh, std::vector<int>* deleteNumbers, int & newpCount);

};