#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <iostream>


#define RAD 0.0174532863
#define SQRT3 1.73205081
#define THREADS_PER_BLOCK 4

__global__ void kernel_generate_mesh(float3 *c, int4 *tet,
	unsigned int nX, unsigned int nY, unsigned int nZ,
	float offsetX, float offsetY, float offsetZ, float edgeLen)
{
	// Optimal angle //
	float angle = 54.7356f;

	float cosa = cos(angle * RAD);
	float sina = sin(angle * RAD);

	float offset_co = edgeLen * cosa * 0.5;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int iz = index / nY;
	int iy = index % nY;


	for (int ix = 0; ix < nX; ix++)
	{
		if ((iy < (nY)) && (iz < (nZ)))
		{

			float curX = offsetX +
				(SQRT3 * ix * edgeLen * cosa) // distance
				+ iy * SQRT3 * offset_co  // offset depends on y
				+ iz * SQRT3 * offset_co; // offset depends on Z
			float curY = offsetY +
				iy * 1.5f * edgeLen * cosa  // distance
				+ iz * offset_co; // offset depends on y
			float curZ = offsetZ +
				iz * edgeLen * sina;

			float3 tmp = { curX, curY, curZ };

			memcpy(&c[ix + iy*nX + iz*nX*nY], &tmp, sizeof(float3));


			if ((ix < (nX - 1)) && (iy < (nY - 1)) && (iz < (nZ - 1)))
			{
				int index1 =
					iz * nX*nY //Z loop
					+ iy * nX // Y loop 
					+ ix; // X loop

				int index2 =
					iz * nX*nY //Z loop
					+ iy * nX // Y loop 
					+ ix + 1; // X loop

				int index3 =
					iz* nX*nY //Z loop
					+ iy * nX // Y loop 
					+ ix + nX; // X loop

				int index4 =
					iz* nX*nY //Z loop
					+ iy * nX // Y loop 
					+ ix + nX * nY; // X loop

				int index5 =
					iz * nX*nY //Z loop
					+ iy * nX // Y loop 
					+ ix + nX + 1; // X loop

				int index6 =
					iz * nX*nY //Z loop
					+ iy * nX // Y loop 
					+ ix + nX*nY + 1; // X loop

				int index7 =
					iz * nX*nY //Z loop
					+ iy * nX // Y loop 
					+ ix + nX*nY + nX; // X loop


				int4 tmp1 = { index1, index2, index3, index4 };
				int4 tmp2 = { index3, index2, index6, index4 };
				int4 tmp3 = { index2, index5, index3, index6 };
				int4 tmp4 = { index4, index6, index7, index3 };
				int4 tmp5 = { index3, index6, index5, index7 };



				//regular tetra
				memcpy(&tet[(ix + iy*(nX - 1) + iz*(nY - 1)*(nX - 1)) * 5], &tmp1, sizeof(int4));

				//first from octa
				memcpy(&tet[(ix + iy*(nX - 1) + iz*(nY - 1)*(nX - 1)) * 5 + 1], &tmp2, sizeof(int4));

				//second from octa
				memcpy(&tet[(ix + iy*(nX - 1) + iz*(nY - 1)*(nX - 1)) * 5 + 2], &tmp3, sizeof(int4));

				//third from octa
				memcpy(&tet[(ix + iy*(nX - 1) + iz*(nY - 1)*(nX - 1)) * 5 + 3], &tmp4, sizeof(int4));

				//fourth from octa
				memcpy(&tet[(ix + iy*(nX - 1) + iz*(nY - 1)*(nX - 1)) * 5 + 4], &tmp5, sizeof(int4));
			}
		}

	}
}

cudaError_t genMeshWithCuda(float3* &dev_points, int4* &dev_tetra,
	unsigned int nX, unsigned int nY, unsigned int nZ,
	float offsetX, float offsetY, float offsetZ,
	float edgeLen)
{
	cudaError_t cudaStatus;
	int pSize = nX * nY * nZ;
	int tetraSize = (nX - 1)*(nY - 1)*(nZ - 1) * 5;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_points, pSize * sizeof(float3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on points!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_tetra, tetraSize * sizeof(int4));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed on tetras!");
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.
	kernel_generate_mesh <<< (nZ*nY / THREADS_PER_BLOCK) + THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
		(dev_points, dev_tetra, nX, nY, nZ, offsetX, offsetY, offsetZ, edgeLen);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "generate_mesh launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching add_points!\n", cudaStatus);
		goto Error;
	}

	return cudaStatus;
Error:
	cudaFree(dev_points);
	cudaFree(dev_tetra);

	return cudaStatus;
}

cudaError_t copyMeshFromGPU(float3* points, float3* dev_points, int pCount,
	int4* tetra, int4* dev_tetra, int tCount)
{
	cudaError_t cudaStatus;
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, pCount * sizeof(float3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy points failed! \n");
		printf("CUDA Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}
	cudaStatus = cudaMemcpy(tetra, dev_tetra, tCount * sizeof(int4), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy tetrahedeons failed! \n");
		printf("CUDA Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}
	return cudaStatus;
Error:
	cudaFree(dev_points);
	cudaFree(dev_tetra);

	return cudaStatus;
}