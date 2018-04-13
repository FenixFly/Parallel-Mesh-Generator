#pragma once
#include "cuda_runtime.h"
#include "helper_math.cuh"
#include <stdio.h>
#include "common.h"

__host__ __device__ int hasIntersection(float3 point, float3 p1, float3 p2, float3 p3)
{
	float3 direction = make_float3(0.0f, 0.0f, 1.0f);

	float3 A = p2 - p1;
	float3 B = p3 - p1;

	float3 N = cross(A, B);
	N = normalize(N);

	// Step 1: finding P
	// check if ray and plane are parallel?
	float NdotRayDirection = dot(N, direction);
	if (abs(NdotRayDirection) < 0.0001)
		return 0; // they are parallel so they don't intersect!

	float d = dot(N, p1);
	float t = -(dot(N, point) - d) / NdotRayDirection;
	if (t < 0)
		return 0;// the triangle is behind
	float3 P = point + t * direction;
	// Step 2: inside-outside test //

	// vector perpendicular to triangle's plane
	// edge 1
	float3 edge1 = p2 - p1;
	float3 VP1 = P - p1;
	float3 C1 = cross(edge1, VP1);
	if (dot(N, C1) < 0)
		return 0; // P is on the right side
					  // edge 2
	float3 edge2 = p3 - p2;
	float3 VP2 = P - p2;
	float3 C2 = cross(edge2, VP2);
	if (dot(N, C2) < 0)
		return 0; // P is on the right side
					  // edge 3
	float3 edge3 = p1 - p3;
	float3 VP3 = P - p3;
	float3 C3 = cross(edge3, VP3);
	if (dot(N, C3) < 0)
		return 0; // P is on the right side
					  // P inside triangle
	return 1;
}

__global__ void calcPipKernel(bool * inside, float3 * points, int pCount, float3 * triangles, int tCount)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int count = 0;
	for (int i = 0; i < tCount; i++)
		count += hasIntersection(
			points[x],
			triangles[0 + i * 3],
			triangles[1 + i * 3],
			triangles[2 + i * 3]);
	if (count % 2 == 0)
		inside[x] = false;
	else
		inside[x] = true;

}

cudaError_t calcIntersectionCuda(
	bool * inside, float3 * points, unsigned int pointsCount,
	float3 * triangles, unsigned int trianglesCount,
	float &timeWithCopy, float &timeWithoutCopy)
{
	float3 *dev_points = 0;
	float3 *dev_triags = 0;
	bool *dev_result = 0;
	cudaError_t cudaStatus;
	cudaEvent_t startWithCopy, stopWithCopy,
		startWithoutCopy, stopWithoutCopy;
	cudaEventCreate(&startWithCopy);
	cudaEventCreate(&stopWithCopy);
	cudaEventCreate(&startWithoutCopy);
	cudaEventCreate(&stopWithoutCopy);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed? \n");
		goto Error;
	}

	// Allocate GPU buffers for points
	cudaStatus = cudaMalloc((void**)&dev_points, pointsCount * sizeof(float3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! \n");
		goto Error;
	}

	// Allocate GPU buffers for results
	cudaStatus = cudaMalloc((void**)&dev_result, pointsCount * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! \n");
		goto Error;
	}

	// Allocate GPU buffers for triangles
	cudaStatus = cudaMalloc((void**)&dev_triags, trianglesCount * 3 * sizeof(float3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! \n");
		goto Error;
	}

	cudaEventRecord(startWithCopy);

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points,
		pointsCount * sizeof(float3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 1  failed! \n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_triags, triangles,
		trianglesCount * 3 * sizeof(float3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 2 failed! \n");
		goto Error;
	}

	cudaEventRecord(startWithoutCopy);


	const int block_size = 512;
	int num_blocks = pointsCount / block_size;

	// Launch a kernel on the GPU with one thread for each element.
	calcPipKernel << < num_blocks, block_size >> >
		(dev_result, dev_points, pointsCount, dev_triags, trianglesCount);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
			"cudaDeviceSynchronize returned error code %d after launching calcPipKernel! \n",
			cudaStatus);
		goto Error;
	}

	cudaEventRecord(stopWithoutCopy);
	///cudaEventSynchronize(stopWithoutCopy);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(inside, dev_result, pointsCount * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 3 failed! \n");
		printf("CUDA Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}

	cudaEventRecord(stopWithCopy);
	cudaEventSynchronize(stopWithCopy);

	cudaEventElapsedTime(&timeWithCopy, startWithCopy, stopWithCopy);
	cudaEventElapsedTime(&timeWithoutCopy, startWithoutCopy, stopWithoutCopy);

	cudaEventDestroy(startWithCopy);
	cudaEventDestroy(startWithoutCopy);
	cudaEventDestroy(stopWithCopy);
	cudaEventDestroy(stopWithoutCopy);

	cudaFree(dev_triags);
	return cudaStatus;

Error:
	cudaFree(dev_points);
	cudaFree(dev_triags);
	cudaFree(dev_result);

	return cudaStatus;
}

cudaError_t calcIntersectionCuda2(
	bool * inside, float3 * dev_points, unsigned int pointsCount,
	float3 * triangles, unsigned int trianglesCount,
	float &timeWithCopy, float &timeWithoutCopy)
{
	float3 *dev_triags = 0;
	bool *dev_result = 0;
	cudaError_t cudaStatus;
	cudaEvent_t startWithCopy, stopWithCopy,
		startWithoutCopy, stopWithoutCopy;
	cudaEventCreate(&startWithCopy);
	cudaEventCreate(&stopWithCopy);
	cudaEventCreate(&startWithoutCopy);
	cudaEventCreate(&stopWithoutCopy);

	// Allocate GPU buffers for results
	cudaStatus = cudaMalloc((void**)&dev_result, pointsCount * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! \n");
		goto Error;
	}

	// Allocate GPU buffers for triangles
	cudaStatus = cudaMalloc((void**)&dev_triags, trianglesCount * 3 * sizeof(float3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! \n");
		goto Error;
	}

	cudaEventRecord(startWithCopy);

	// Copy STL triangles
	cudaStatus = cudaMemcpy(dev_triags, triangles,
		trianglesCount * 3 * sizeof(float3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 2 failed! \n");
		goto Error;
	}

	cudaEventRecord(startWithoutCopy);


	const int block_size = 512;
	int num_blocks = pointsCount / block_size;

	// Launch a kernel on the GPU with one thread for each element.
	calcPipKernel <<< num_blocks, block_size >>>
		(dev_result, dev_points, pointsCount, dev_triags, trianglesCount);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
			"cudaDeviceSynchronize returned error code %d after launching calcPipKernel! \n",
			cudaStatus);
		goto Error;
	}

	cudaEventRecord(stopWithoutCopy);
	///cudaEventSynchronize(stopWithoutCopy);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(inside, dev_result, pointsCount * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 3 failed! \n");
		printf("CUDA Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}

	cudaEventRecord(stopWithCopy);
	cudaEventSynchronize(stopWithCopy);

	cudaEventElapsedTime(&timeWithCopy, startWithCopy, stopWithCopy);
	cudaEventElapsedTime(&timeWithoutCopy, startWithoutCopy, stopWithoutCopy);

	cudaEventDestroy(startWithCopy);
	cudaEventDestroy(startWithoutCopy);
	cudaEventDestroy(stopWithCopy);
	cudaEventDestroy(stopWithoutCopy);

	return cudaStatus;

Error:
	cudaFree(dev_points);
	cudaFree(dev_triags);
	cudaFree(dev_result);

	return cudaStatus;
}
