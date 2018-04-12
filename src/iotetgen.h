#pragma once
#include "common.h"
#include "grainmesh.h"
#include "mymesh.h"

namespace grain
{
    int readNodeFile(std::string filename, GrainMesh * mesh);
	int readFaceFile(std::string filename, GrainMesh *mesh);
    int readEleFile(std::string filename, GrainMesh * mesh);
    int saveNodeFile(std::string filename, GrainMesh * mesh);
	int saveFaceFile(std::string filename, GrainMesh * mesh);
    int saveEleFile(std::string filename, GrainMesh * mesh);
	int readNodeFile(std::string filename, MyMesh * mesh);
	int readEleFile(std::string filename, MyMesh * mesh);
	int saveNodeFile(std::string filename, MyMesh * mesh);
	int saveEleFile(std::string filename, MyMesh * mesh);
}

