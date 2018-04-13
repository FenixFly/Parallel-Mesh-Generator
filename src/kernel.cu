/*
* Name: cuda_pip.cuh
* Author  : Evgenii Vasilev
* Created : 26.12.2016
* Description: Input-output, time measure
* Version: 1.0
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_pip.cuh"
#include "cuda_meshgen.cuh"
#include "helper_math.cuh"

#include <stdio.h>
#include <iostream>
#include <fstream>

#include "grainmesh.h"
#include "iotetgen.h"
#include "mystl.h"
#include "mymesh.h"
#include "meshcut.h"
#include "meshsmooth.h"
#include <ctime>

void convertMeshToArrays(grain::GrainMesh* mesh, float3* &vertices, std::vector<float> * tr, float3* &triangles);

int main()
{
	
	int nX = 64, nY = 64, nZ = 64;
	float offX = -17.0f, offY = -7.0f, offZ = -7.0f;
	float edgeLen = 0.2;
	int pCount = nX * nY * nZ;
	int tCount = (nX - 1)*(nY - 1)*(nZ - 1) * 6;

	float3 *dev_points = 0;
	int4 *dev_tetra = 0;

	clock_t  timeMeshGenStart, timeMeshGenEnd, 
		timeMeshMarkStart, timeMeshMarkEnd,
		timeMeshCutStart, timeMeshCutEnd,
		timeMeshSmoothStart, timeMeshSmoothEnd;
	
	// Generate mesh with CUDA //
	timeMeshGenStart = clock();
	cudaError_t cudaStatus = genMeshWithCuda(dev_points, dev_tetra,
		nX, nY, nZ, offX, offY, offZ, edgeLen);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "genMeshWithCuda failed!");
		return 1;
	}
	timeMeshGenEnd = clock();
	
	
	/// Copy Mesh from GPU and save in file //

	//float3 *points = new float3[pCount];
	//int4 *tetra = new int4[tCount];
	//copyMeshFromGPU(points, dev_points, pCount,
	//	tetra, dev_tetra, tCount);

	//// Save mesh //
	//MyMesh mymesh;
	//mymesh.mPoints = points;
	//mymesh.mPointsCount = pCount;
	//short* pLabels = new short[pCount];
	//for (int i = 0; i < pCount; i++)
	//	pLabels[i] = 0;
	//mymesh.mPointLabels = pLabels;

	//mymesh.mTetra = tetra;
	//mymesh.mTetraCount = tCount;
	//short* tLabels = new short[tCount];
	//for (int i = 0; i < tCount; i++)
	//	tLabels[i] = 0;
	//mymesh.mTetraLabels = tLabels;

	//grain::saveNodeFile("E:/Data/STL/results/gpu.node", &mymesh);
	//grain::saveEleFile("E:/Data/STL/results/gpu.ele", &mymesh);

	
	float timeWithCopy, timeWithoutCopy;
	MySTL stl;
	stl.readSTL("E:/Data/STL/CyberheartModel/00_heart_shell.stl");
	float3* mystl = new float3[stl.trigs.size()/3];
	for (int i = 0; i < stl.trigs.size() / 3; i++)
	{
		mystl[i].x = stl.trigs[3 * i + 0];
		mystl[i].y = stl.trigs[3 * i + 1];
		mystl[i].z = stl.trigs[3 * i + 2];
	}
	bool * result = new bool[pCount];
	timeMeshMarkStart = clock();
	// Mark mesh with CUDA //
	cudaStatus = calcIntersectionCuda2(result, 
		dev_points, pCount,
		mystl, stl.trigs.size()/9,
		timeWithCopy, timeWithoutCopy);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcIntersectionCuda failed! \n");
		return 1;
	}
	timeMeshMarkEnd = clock();
	
	// Generate mesh with labels //
	short* resvec = new short[pCount];
	for (int i = 0; i < pCount; i++)
	{
		int val = 999;
		if (result[i] == true)
			val = 0;
		resvec[i] = val;
	}
	
	/// Copy Mesh from GPU and save in file //
	
	float3 *points = new float3[pCount];
	int4 *tetra = new int4[tCount];
	copyMeshFromGPU(points, dev_points, nX*nY*nZ,
		tetra, dev_tetra, tCount);


	MyMesh mymesh;
	mymesh.mPoints = points;
	mymesh.mPointsCount = pCount;
	mymesh.mPointLabels = resvec;

	mymesh.mTetra = tetra;
	mymesh.mTetraCount = tCount;
	short* tLabels = new short[tCount];
	for (int i = 0; i < tCount; i++)
		tLabels[i] = 0;
	mymesh.mTetraLabels = tLabels;
	
	
	//grain::saveNodeFile("E:/Data/STL/results/gpu2.node", &mymesh);
	//grain::saveEleFile("E:/Data/STL/results/gpu2.ele", &mymesh);
	
	


	/*
	// Convert to mymesh //
	MyMesh mymesh;

	short* pLabels = new short[mesh.getVerticesCount()];
	for (int i = 0; i < mesh.getVerticesCount(); i++)
		pLabels[i] = resvec[i];
	mymesh.mPoints = verticesOfMesh;
	mymesh.mPointsCount = mesh.getVerticesCount();
	mymesh.mPointLabels = pLabels;

	std::vector<vec4i> * tt = mesh.getTetra();
	int4* t = new int4[mesh.getTetraCount()];
	for (int i = 0; i < mesh.getTetraCount(); i++)
	{
		memcpy(&t[i], &tt->at(i), sizeof(vec4i));
	}
	short* tLabels = new short[mesh.getTetraCount()];
	for (int i = 0; i < mesh.getTetraCount(); i++)
		tLabels[i] = 0;
	mymesh.mTetra = t;
	mymesh.mTetraCount = mesh.getTetraCount();
	mymesh.mTetraLabels = tLabels;
	
	grain::saveNodeFile("beforeCut.node", &mymesh);
	grain::saveEleFile("beforeCut.ele", &mymesh);
	

	MyMesh mymesh;
	grain::readNodeFile("E:/Data/STL/results/beforeCut.node", &mymesh);
	grain::readEleFile("E:/Data/STL/results/beforeCut.ele", &mymesh);
	*/

	

	// Cutting //

	timeMeshCutStart = clock();
	MeshCut cut;
	cut.cutMeshMarkedVertices(&mymesh);
	timeMeshCutEnd = clock();
	
	
	//grain::saveNodeFile("E:/Data/STL/results/afterCut.node", &mymesh);
	//grain::saveEleFile("E:/Data/STL/results/afterCut.ele", &mymesh);
	
	// Smoothing //

	
	//MyMesh mymesh;
	//grain::readNodeFile("E:/Data/STL/results/afterCut.node", &mymesh);
	//grain::readEleFile("E:/Data/STL/results/afterCut.ele", &mymesh);
	//MySTL stl;
	//stl.readSTL("E:/Data/STL/CyberheartModel/00_heart_shell.stl");
	


	timeMeshSmoothStart = clock();
	MeshSmooth smooth;
	smooth.edgelen = edgeLen;
	smooth.smoothMesh(&mymesh, &stl);
	timeMeshSmoothEnd = clock();



	grain::saveNodeFile("E:/Data/STL/results/afterSmooth.node", &mymesh);
	grain::saveEleFile("E:/Data/STL/results/afterSmooth.ele", &mymesh);
	
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	/*cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed! \n");
		return 1;
	}*/

	std::ofstream fout("result.txt");
	fout << "Time mesh generate " << (float)(timeMeshGenEnd - timeMeshGenStart) / CLK_TCK << "\n"
		<< "Time mesh mark " << (float)(timeMeshMarkEnd - timeMeshMarkStart) / CLK_TCK << "\n"
		<< " Time mesh cut " << (float)(timeMeshCutEnd - timeMeshCutStart) / CLK_TCK << "\n"
		<< " Time mesh smooth " << (float)(timeMeshSmoothEnd - timeMeshSmoothStart) / CLK_TCK << "\n";
	fout.close();


	return 0;
}

void convertMeshToArrays(grain::GrainMesh* mesh, float3* &vertices, std::vector<float> * tr, float3* &triangles)
{
	std::vector<vec3d>* vert = mesh->getVertices();
	vertices = new float3[mesh->getVerticesCount()];
	for (int i = 0; i < mesh->getVerticesCount(); i++)
	{
		vertices[i].x = vert->at(i).x;
		vertices[i].y = vert->at(i).y;
		vertices[i].z = vert->at(i).z;
	}
	triangles = new float3[tr->size() / 3];
	for (int i = 0; i <tr->size() / 3; i++)
	{
		triangles[i].x = tr->at(i * 3 + 0);
		triangles[i].y = tr->at(i * 3 + 1);
		triangles[i].z = tr->at(i * 3 + 2);
	}
}