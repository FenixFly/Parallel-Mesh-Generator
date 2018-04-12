#pragma once
#include <iostream>
#include "common.h"
#include "helper_math.cuh"
#include <vector>
#include <set>
#include "mymesh.h"

using std::set;

class MeshCut 
{
public:
	std::vector<std::set<int>*>* adjenctionList = nullptr;
	void cutMeshMarkedVertices(MyMesh * mesh);
private:

	void createRelationshipListInternal(
		MyMesh* mesh,
		std::vector<int2>* &vertRelations,
		int &newPointsNumber);

	void createRelationshipListSaveOneLayer(
		MyMesh* mesh,
		std::vector<int2>* &vertRelations,
		int &newPointsNumber);

	void fillAdjunctionsList(MyMesh * mesh, std::vector<std::set<int>*> * &adjList);

public:
	void deleteNonRelationPoints(
		MyMesh* mesh,
		std::vector<int2>* list,
		int newPointsNumber);

	void deleteNonRelationTetra(
		MyMesh* mesh,
		std::vector<int2>* list);
};