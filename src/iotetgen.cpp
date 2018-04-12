#include "iotetgen.h"
#include <fstream>

int grain::readNodeFile(std::string filename, GrainMesh * mesh)
{
	std::ifstream fin(filename, std::ifstream::in);
	int pCount = 0;
	int trash = 0;
	fin >> pCount;
	fin >> trash;
	fin >> trash;
	fin >> trash;
	fin >> trash;
	std::vector<vec3d> newPoints(pCount);
	std::vector<char> newPLabels(pCount);
	for (int i = 0; i < pCount; i++)
	{
		fin >> trash
			>> newPoints[i].x
			>> newPoints[i].y
			>> newPoints[i].z
			>> newPLabels[i];
	}
	fin.close();
	mesh->setVertices(newPoints);
	mesh->setVerticesLabels(newPLabels);
	return 0;
}

int grain::readFaceFile(std::string filename, GrainMesh *mesh)
{
	std::ifstream fin(filename, std::ifstream::in);
	int fCount = 0;
	int trash = 0;
	bool marker = false;

	fin >> fCount;
	fin >> trash;

	if (trash == 1) // Have boundary marker
		marker = true;
	std::vector<vec3i> newFacet(fCount);
	std::vector<char> newFLabels(fCount);

	for (int i = 0; i < fCount; i++)
	{
		fin >> trash
			>> newFacet[i].x
			>> newFacet[i].y
			>> newFacet[i].z;
		if (marker)
			fin >> newFLabels[i];

		// Vertices in file start from 1, not from 0
		newFacet[i].x--;
		newFacet[i].y--;
		newFacet[i].z--;
	}
	fin.close();
	mesh->setTriangles(newFacet);
	if (marker)
		mesh->setTrianglesLabels(newFLabels);

	return 0;
}

int grain::readEleFile(std::string filename, GrainMesh *mesh)
{
	std::ifstream fin(filename, std::ifstream::in);
	int tCount = 0;
	int trash = 0;
	fin >> tCount;
	fin >> trash;
	fin >> trash;
	std::vector<vec4i> newTetra(tCount);
	std::vector<char> newTLabels(tCount);
	for (int i = 0; i < tCount; i++)
	{
		fin >> trash
			>> newTetra[i].x
			>> newTetra[i].y
			>> newTetra[i].z
			>> newTetra[i].w
			>> newTLabels[i];
	}
	fin.close();
	mesh->setTetra(newTetra);
	mesh->setTetraLabels(newTLabels);
	return 0;
}

int grain::saveNodeFile(std::string filename, GrainMesh *mesh)
{
	int mPointsCount = mesh->getVerticesCount();
	std::vector<vec3d>* mPoints = mesh->getVertices();
	std::vector<char>* mPointsLabels = mesh->getVerticesLabels();
	bool marker = false;
	if (mPointsLabels->size() == mPoints->size())
		marker = true;

	std::ofstream fout(filename);
	fout << mPointsCount << " " << 3 << " " << 0 << " " << 0 << " " << 0 << "\n";
	for (int i = 0; i < mPointsCount; i++)
	{
		fout << i + 1 << " "
			<< mPoints->at(i).x << " "
			<< mPoints->at(i).y << " "
			<< mPoints->at(i).z;
		if (marker)
			fout << " " << (int)mPointsLabels->at(i) << "\n";
		else
			fout << "\n";
	}
	fout.close();
	return 0;
}

int grain::saveFaceFile(std::string filename, GrainMesh * mesh)
{
	int mTrianglesCount = mesh->getTrianglesCount();
	std::vector<vec3i>* mTriangles = mesh->getTriangles();
	std::vector<char>* mTrianglesLabels = mesh->getTrianglesLabels();
	bool marker = false;
	if (mTrianglesLabels->size() == mTriangles->size())
		marker = true;

	std::ofstream fout(filename);
	fout << mTrianglesCount;
	if (marker)
		fout << " 1\n";
	else
		fout << " 0\n";
	for (int i = 0; i < mTrianglesCount; i++)
	{
		fout << i << " "
			<< mTriangles->at(i).x << " "
			<< mTriangles->at(i).y << " "
			<< mTriangles->at(i).z;
		if (marker)
			fout << " " << (int)mTrianglesLabels->at(i) << "\n";
		else
			fout << "\n";
	}
	fout.close();

	return 0;
}

int grain::saveEleFile(std::string filename, GrainMesh *mesh)
{
	int mTetraCount = mesh->getTetraCount();
	std::vector<vec4i>* mTetra = mesh->getTetra();
	std::vector<char>* mTetraLabels = mesh->getTetraLabels();
	bool marker = false;
	if (mTetraLabels->size() == mTetra->size())
		marker = true;

	std::ofstream fout(filename);
	fout << mTetraCount << " " << 4;
	if (marker)
		fout << " 1\n";
	else
		fout << " 0\n";
	fout.flush();
	for (int i = 0; i < mTetraCount; i++)
	{
		fout << i << " "
			<< mTetra->at(i).x << " "
			<< mTetra->at(i).y << " "
			<< mTetra->at(i).z << " "
			<< mTetra->at(i).w;
		if (marker)
			fout << " " << (int)mTetraLabels->at(i) << "\n";
		else
			fout << "\n";
	}
	fout.close();
	return 0;
}

int grain::readNodeFile(std::string filename, MyMesh * mesh)
{
	std::ifstream fin(filename, std::ifstream::in);
	int pCount = 0;
	int trash = 0;
	fin >> pCount;
	fin >> trash;
	fin >> trash;
	fin >> trash;
	fin >> trash;

	float3* newPoints = new float3[pCount];
	short* newPLabels = new short[pCount];
	for (int i = 0; i < pCount; i++)
	{
		fin >> trash
			>> newPoints[i].x
			>> newPoints[i].y
			>> newPoints[i].z
			>> newPLabels[i];
	}
	fin.close();
	mesh->mPoints = newPoints;
	mesh->mPointsCount = pCount;
	mesh->mPointLabels = newPLabels;
	return 0;
}

int grain::readEleFile(std::string filename, MyMesh * mesh)
{
	std::ifstream fin(filename, std::ifstream::in);
	int tCount = 0;
	int trash = 0;
	fin >> tCount;
	fin >> trash;
	fin >> trash;
	int4* newTetra = new int4[tCount];
	short* newTLabels = new short[tCount];
	for (int i = 0; i < tCount; i++)
	{
		fin >> trash
			>> newTetra[i].x
			>> newTetra[i].y
			>> newTetra[i].z
			>> newTetra[i].w
			>> newTLabels[i];
	}
	fin.close();
	mesh->mTetra = newTetra;
	mesh->mTetraCount = tCount;
	mesh->mTetraLabels = newTLabels;
	return 0;
}

int grain::saveNodeFile(std::string filename, MyMesh * mesh)
{
	int mPointsCount = mesh->mPointsCount;
	float3* mPoints = mesh->mPoints;
	short* mPointsLabels = mesh->mPointLabels;
	
	bool marker = false;
	if (mPointsLabels != nullptr)
		marker = true;

	std::ofstream fout(filename);
	fout << mPointsCount << " " << 3 << " " << 0 << " " << 0 << " " << 0 << "\n";
	for (int i = 0; i < mPointsCount; i++)
	{
		fout << i + 1 << " "
			<< mPoints[i].x << " "
			<< mPoints[i].y << " "
			<< mPoints[i].z;
		if (marker)
			fout << " " << mPointsLabels[i] << "\n";
		else
			fout << "\n";
	}
	fout.close();
	return 0;
}

int grain::saveEleFile(std::string filename, MyMesh * mesh)
{
	int mTetraCount = mesh->mTetraCount;
	int4* mTetra = mesh->mTetra;
	short* mTetraLabels = mesh->mTetraLabels;
	bool marker = false;
	if (mTetraLabels != nullptr)
		marker = true;

	std::ofstream fout(filename);
	fout << mTetraCount << " " << 4;
	if (marker)
		fout << " 1\n";
	else
		fout << " 0\n";
	fout.flush();
	for (int i = 0; i < mTetraCount; i++)
	{
		fout << i << " "
			<< mTetra[i].x << " "
			<< mTetra[i].y << " "
			<< mTetra[i].z << " "
			<< mTetra[i].w;
		if (marker)
			fout << " " << mTetraLabels[i] << "\n";
		else
			fout << "\n";
	}
	fout.close();
	return 0;
}
