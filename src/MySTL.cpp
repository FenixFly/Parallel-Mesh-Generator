/*
* Name: MySTL.cpp
* Author  : Evgenii Vasilev
* Created : 26.12.2016
* Description: Реализация класса MySTL
*              класс для хранения данных, прочитанных из .stl файла
* Version: 1.0
*/
#include "mystl.h"
#include <iostream>
#include <fstream>
#include <math.h>


bool MySTL::isBinary(std::string fileName)
{
	std::ifstream fin;
	fin.open(fileName);
	char str[80];
	fin.getline(str, 80);
	fin.close();
	char* pos = strstr(str, "solid");
	char* pos2 = strstr(str, "facet");

	if ((pos == nullptr) && (pos == nullptr))
		return true;
	return false;
}

int MySTL::load(std::string fileName)
{
	std::ifstream ist(fileName);

	
	char tmpstr[100];
	int countFacet = 0;
	int countVertex = 0;
	while (ist.good())
	{
		ist >> tmpstr;

		int n = strlen(tmpstr);
		for (int i = 0; i < n; i++)
			tmpstr[i] = tolower(tmpstr[i]);
				
		if (strcmp(tmpstr, "vertex") == 0)
		{
			float x;
			float y;
			float z;
			ist >> x >> y >> z;
			trigs.push_back(x);
			trigs.push_back(y);
			trigs.push_back(z);
		}
	}
	ist.close();
	return 0;
}

int MySTL::loadBinary(std::string fileName)
{
	return 1;
}

void MySTL::saveShellInTXT(std::string filepath)
{
	int trigsCount = trigs.size() / 9;
	std::ofstream fout(filepath.c_str());
	fout << trigsCount << "\n";
	for (int i = 0; i < trigsCount; i++)
		fout << trigs.at(i * 9 + 0) << " "
		<< trigs.at(i * 9 + 1) << " "
		<< trigs.at(i * 9 + 2) << " "
		<< trigs.at(i * 9 + 3) << " "
		<< trigs.at(i * 9 + 4) << " "
		<< trigs.at(i * 9 + 5) << " "
		<< trigs.at(i * 9 + 6) << " "
		<< trigs.at(i * 9 + 7) << " "
		<< trigs.at(i * 9 + 8) << "\n";
	fout.close();
}

int MySTL::readSTL(std::string fileName)
{
	int result = 0;
	if (isBinary(fileName))
		result = loadBinary(fileName);
	else
		result = load(fileName);
	int p = 1 + std::fmin(fileName.find_last_of("/"), fileName.find_last_of("\\"));
	myname = fileName.substr(p, fileName.length() - p - 4);
	return result;
}

MySTL::MySTL()
{

}


MySTL::~MySTL()
{

}
