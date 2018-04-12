#include "meshcut.h"

void MeshCut::cutMeshMarkedVertices(MyMesh* mesh)
{
	fillAdjunctionsList(mesh, adjenctionList);

	std::vector<int2>* vertRelations = new std::vector<int2>();
	int newPointsNumber = 0;

	//Находим соответствие номеров вершин до и после удаления
	//createRelationshipListInternal(mesh, vertRelations, newPointsNumber);
	createRelationshipListSaveOneLayer(mesh, vertRelations, newPointsNumber);

	//Удаляем помеченные вершины из списка по найденному соответствию
	deleteNonRelationPoints(mesh, vertRelations, newPointsNumber);

	//Переименовываем номера вершин в тетраэдрах и удаляем тетраэдры, в которых
	//есть удаленные вершины
	deleteNonRelationTetra(mesh, vertRelations);
}

void MeshCut::createRelationshipListInternal(
	MyMesh* mesh,
	std::vector<int2>* &vertRelations,
	int &newPointsNumber)
{
	newPointsNumber = 0;
	int pCount = mesh->mPointsCount;
	for (int i = 0; i < pCount; i++)
	{
		if (mesh->mPointLabels[i] > 100)
			vertRelations->push_back(make_int2(i, -1));
		else
		{
			vertRelations->push_back(make_int2(i, newPointsNumber));
			newPointsNumber++;
		}
	}
}

void MeshCut::createRelationshipListSaveOneLayer(MyMesh * mesh, std::vector<int2>*& vertRelations, int & newPointsNumber)
{
	float3* p = mesh->mPoints;
	short * pLabels = mesh->mPointLabels;
	int pCount = mesh->mPointsCount;

	newPointsNumber = 0;
	for (int i = 0; i < pCount; i++)
	{
		/*
		Удаляем вершину только если она помечена и все ее соседи помечены
		*/
		set<int>* tmp = adjenctionList->at(i);
		bool hasUnmarkedNeighbour = false;
		for (std::set<int>::iterator it = tmp->begin(); it != tmp->end(); ++it)
		{
			int val = *it;
			if (pLabels[val] == 0)
			{
				hasUnmarkedNeighbour = true;
				break;
			}
		}

		if ((pLabels[i] > 100) && (hasUnmarkedNeighbour == false))
		{
			vertRelations->push_back(make_int2(i, -1));
		}
		else
		{
			vertRelations->push_back(make_int2(i, newPointsNumber));
			newPointsNumber++;
		}
	}
}

void MeshCut::deleteNonRelationPoints(
	MyMesh* mesh,
	std::vector<int2>* list,
	int newPointsNumber)
{
	int t = 2143324;
	// Old points //
	float3* p = mesh->mPoints;
	short* pLabels = mesh->mPointLabels;
	int pCount = mesh->mPointsCount;

	// New points //
	int newpCount = newPointsNumber;
	float3 * newp = new float3[newpCount];
	short* newpLabels = new short[newpCount];

	for (int i = 0; i < pCount; i++)
	{
		if (i == 2768)
		{
			t = i* i;
		}

		int pos = list->at(i).y;
		if (pos != -1)
		{
			memcpy(&newp[pos], &p[i], sizeof(float3));
			newpLabels[pos] = pLabels[i];
		}
	}
	delete[] p;
	delete[] pLabels;

	mesh->mPointsCount = newpCount;
	mesh->mPoints = newp;
	mesh->mPointLabels = newpLabels;
}

void MeshCut::deleteNonRelationTetra(
	MyMesh* mesh,
	std::vector<int2>* list)
{
	// Old tetra //
	int4* t = mesh->mTetra;
	short* tLabels = mesh->mTetraLabels;
	int tCount = mesh->mTetraCount;
	// New Tetra //
	int4* newt;
	short* newtLabels;
	int newtCount = 0;

	//Check we can delete it
	for (int i = 0; i < tCount; i++)
	{
		int oldvalue, newvalue;
		bool validtetra = true;
		
		oldvalue = t[i].x;
		newvalue = list->at(oldvalue).y;
		t[newtCount].x = newvalue;
		tLabels[newtCount] = tLabels[i];
		if (newvalue < 0)
			validtetra = false;

		oldvalue = t[i].y;
		newvalue = list->at(oldvalue).y;
		t[newtCount].y = newvalue;
		tLabels[newtCount] = tLabels[i];
		if (newvalue < 0)
			validtetra = false;

		oldvalue = t[i].z;
		newvalue = list->at(oldvalue).y;
		t[newtCount].z = newvalue;
		tLabels[newtCount] = tLabels[i];
		if (newvalue < 0)
			validtetra = false;

		oldvalue = t[i].w;
		newvalue = list->at(oldvalue).y;
		t[newtCount].w = newvalue;
		tLabels[newtCount] = tLabels[i];
		if (newvalue < 0)
			validtetra = false;

		if (validtetra == true)
			newtCount++;
	}

	newt = new int4[newtCount];
	newtLabels = new short[newtCount];
	memcpy(newt, t, newtCount * sizeof(int4));
	memcpy(newtLabels, tLabels, newtCount * sizeof(short));

	delete[] t;
	delete[] tLabels;

	mesh->mTetraCount = newtCount;
	mesh->mTetra = newt;
	mesh->mTetraLabels = newtLabels;
}

void MeshCut::fillAdjunctionsList(MyMesh * mesh, std::vector<std::set<int>*>* &adjList)
{
	if (adjList == nullptr)
		adjList = new std::vector<set<int>*>();

	//points
	float3* p = mesh->mPoints;
	short * pLabels = mesh->mPointLabels;
	int pCount = mesh->mPointsCount;
	//tetras
	int4* v = mesh->mTetra;
	int vCount = mesh->mTetraCount;

	for (int i = 0; i < pCount; i++)
		adjList->push_back(new set<int>());

	for (int i = 0; i < vCount; i++)
	{
		int pNum1 = v[i].x;
		set<int>* tmp1 = adjList->at(pNum1);
		tmp1->insert(v[i].y);
		tmp1->insert(v[i].z);
		tmp1->insert(v[i].w);

		int pNum2 = v[i].y;
		set<int>* tmp2 = adjList->at(pNum2);
		tmp2->insert(v[i].x);
		tmp2->insert(v[i].z);
		tmp2->insert(v[i].w);

		int pNum3 = v[i].z;
		set<int>* tmp3 = adjList->at(pNum3);
		tmp3->insert(v[i].x);
		tmp3->insert(v[i].y);
		tmp3->insert(v[i].w);

		int pNum4 = v[i].w;
		set<int>* tmp4 = adjList->at(pNum4);
		tmp4->insert(v[i].x);
		tmp4->insert(v[i].y);
		tmp4->insert(v[i].z);
	}
}
