/*
* Name: cuda_pip.cuh
* Author  : Evgenii Vasilev
* Created : 05.04.2016
* Description: Main function in mesher program
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
	///
	/// Change parameters here ///
	///

	// Filename of heart shell // 
	std::string fileHeartStl;
	fileHeartStl = "E:/Data/STL/cube3.stl"; 
	//fileHeartStl = "E:/Data/STL/CyberheartModel/00_heart_shell.stl";
	// Folder to store mesh results //
	std::string folderpath = "E:/Data/STL/results/";


	// Set mesh parameters //

	//int nX = 100, nY = 150, nZ = 150;
	//float offX = -16.0f, offY = -7.3f, offZ = -7.0f;
	//float edgeLen = 0.11;
	int nX = 75, nY = 75, nZ = 75;
	float offX = -18.5f, offY = -7.f, offZ = -7.5f;
	float edgeLen = 0.2;

	bool generateMeshGPU = true;
	bool saveMeshAfterGenerate = false;
	bool loadMeshBeforeMark = false;
	bool markMeshGPU = true;
	bool saveMeshAfterMark = false;
	bool loadMeshBeforeCut = false;
	bool cutMesh = true;
	bool saveMeshAfterCut = true;
	bool loadMeshBeforeSmooth = false;
	bool smoothMesh = true;
	bool saveMeshAfterSmooth = true;

	///
	/// Change parameters here  ///
	///


	int pCount = nX * nY * nZ;
	int tCount = (nX - 1)*(nY - 1)*(nZ - 1) * 6;
	MyMesh mymesh;
	MySTL stl; stl.readSTL(fileHeartStl);


	float3 *dev_points = 0;
	int4 *dev_tetra = 0;

	clock_t  timeMeshGenStart = 0, timeMeshGenEnd = 0,
		timeMeshMarkStart = 0, timeMeshMarkEnd = 0,
		timeMeshCutStart = 0, timeMeshCutEnd = 0,
		timeMeshSmoothStart = 0, timeMeshSmoothEnd = 0;
	
	// Generate mesh // 
	if (generateMeshGPU == true)
	{
		// Generate mesh with CUDA //
		timeMeshGenStart = clock();
		cudaError_t cudaStatus = genMeshWithCuda(dev_points, dev_tetra,
			nX, nY, nZ, offX, offY, offZ, edgeLen);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "genMeshWithCuda failed!");
			return 1;
		}
		timeMeshGenEnd = clock();
	}
	
	// Save mesh after generate //
	if (saveMeshAfterGenerate == true)
	{
		// Copy Mesh from GPU and save in file //
		float3 *points = new float3[pCount];
		int4 *tetra = new int4[tCount];
		copyMeshFromGPU(points, dev_points, pCount,
			tetra, dev_tetra, tCount);

		// Save mesh //
		MyMesh mymesh;
		mymesh.mPoints = points;
		mymesh.mPointsCount = pCount;
		short* pLabels = new short[pCount];
		for (int i = 0; i < pCount; i++)
			pLabels[i] = 0;
		mymesh.mPointLabels = pLabels;

		mymesh.mTetra = tetra;
		mymesh.mTetraCount = tCount;
		short* tLabels = new short[tCount];
		for (int i = 0; i < tCount; i++)
			tLabels[i] = 0;
		mymesh.mTetraLabels = tLabels;

		grain::saveNodeFile(folderpath + "meshGenerated.node", &mymesh);
		grain::saveEleFile(folderpath + "meshGenerated.ele", &mymesh);
	}
	
	if (loadMeshBeforeMark == true)
	{
		grain::readNodeFile(folderpath + "torscoloredremoved.node", &mymesh);
		//grain::readEleFile(folderpath + "torscoloredremoved.ele", &mymesh);

		pCount = mymesh.mPointsCount;
		//tCount = mymesh.mTetraCount;

		cudaError_t cudaStatus;
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed? \n");
		}
		// Allocate GPU buffers for points
		cudaStatus = cudaMalloc((void**)&dev_points, pCount * sizeof(float3));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed! \n");
		}
		float3* p = mymesh.mPoints;
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_points, p,
			pCount * sizeof(float3), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy 1  failed! \n");
		}

		// Copy input vectors from host memory to GPU buffers.
		//cudaStatus = cudaMemcpy(dev_tetra, mymesh.mTetra,
		//	tCount * sizeof(int4), cudaMemcpyHostToDevice);
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "cudaMemcpy 1  failed! \n");
		//}

	}

	// Mark labels with CUDA // 
	if (markMeshGPU == true)
	{
		float timeWithCopy, timeWithoutCopy;
		float3* mystl = new float3[stl.trigs.size() / 3];
		for (int i = 0; i < stl.trigs.size() / 3; i++)
		{
			mystl[i].x = stl.trigs[3 * i + 0];
			mystl[i].y = stl.trigs[3 * i + 1];
			mystl[i].z = stl.trigs[3 * i + 2];
		}
		bool * result = new bool[pCount];
		timeMeshMarkStart = clock();
		// Mark mesh with CUDA //
		cudaError_t cudaStatus = calcIntersectionCuda2(result,
			dev_points, pCount,
			mystl, stl.trigs.size() / 9,
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
		copyMeshFromGPU(points, dev_points, pCount,
			tetra, dev_tetra, tCount);

		mymesh.mPoints = points;
		mymesh.mPointsCount = pCount;
		mymesh.mPointLabels = resvec;

		mymesh.mTetra = tetra;
		mymesh.mTetraCount = tCount;
		short* tLabels = new short[tCount];
		for (int i = 0; i < tCount; i++)
			tLabels[i] = 0;
		mymesh.mTetraLabels = tLabels;
	}
	
	// Save mesh after mark // 
	if (saveMeshAfterMark == true)
	{
		grain::saveNodeFile(folderpath + "meshMarked.node", &mymesh);
		grain::saveEleFile(folderpath + "meshMarked.ele", &mymesh);
	}

	// Load mesh before cut // 
	if (loadMeshBeforeCut == true)
	{
		grain::readNodeFile(folderpath + "meshMarked.node", &mymesh);
		grain::readEleFile(folderpath + "meshMarked.ele", &mymesh);
	}

	// Mesh cutting //
	if (cutMesh == true)
	{
		timeMeshCutStart = clock();
		MeshCut cut;
		cut.cutMeshMarkedVertices(&mymesh);
		timeMeshCutEnd = clock();
	}
	
	// Save mesh before cut // 
	if (saveMeshAfterCut == true)
	{
		grain::saveNodeFile(folderpath + "afterCut.node", &mymesh);
		grain::saveEleFile(folderpath + "afterCut.ele", &mymesh);
	}

	// Load mesh before smooth // 
	if (loadMeshBeforeSmooth == true)
	{
		grain::readNodeFile(folderpath + "afterCut.node", &mymesh);
		grain::readEleFile(folderpath + "afterCut.ele", &mymesh);
	}
	
	// Smoothing //
	if (smoothMesh == true)
	{
		timeMeshSmoothStart = clock();
		MeshSmooth smooth;
		smooth.edgelen = edgeLen;
		smooth.smoothMesh(&mymesh, &stl);
		timeMeshSmoothEnd = clock();
	}

	// Save smooth after smooth // 
	if (saveMeshAfterSmooth == true)
	{
		grain::saveNodeFile(folderpath + "afterSmooth.node", &mymesh);
		grain::saveEleFile(folderpath + "afterSmooth.ele", &mymesh);
	}
	
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	/*cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed! \n");
		return 1;
	}*/

	std::ofstream fout(folderpath + "result.txt");
	fout << " Time mesh generate " << (float)(timeMeshGenEnd - timeMeshGenStart) / CLK_TCK << "\n"
		<< " Time mesh mark " << (float)(timeMeshMarkEnd - timeMeshMarkStart) / CLK_TCK << "\n"
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