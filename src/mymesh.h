#pragma once
#include "helper_math.cuh"

class MyMesh
{
public:
	MyMesh();
	float3* mPoints;
	short* mPointLabels;
	int mPointsCount;
	int4* mTetra;
	short* mTetraLabels;
	int mTetraCount;
};	