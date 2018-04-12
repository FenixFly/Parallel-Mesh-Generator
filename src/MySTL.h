/*
* Name: MySTL.h
* Author  : Evgenii Vasilev
* Created : 26.12.2016
* Description: ќбъ€вление класса MySTL
*              класс дл€ хранени€ данных, прочитанных из .stl файла
* Version: 1.0
*/
#pragma once
#include <string>
#include <vector>

///TypeDefs
typedef int Point3i[3];
typedef double Point3d[3];
typedef float Point3f[3];
typedef Point3f triangle[3];
///TypeDefs


class MySTL
{
protected:
	bool isBinary(std::string fileName);
	int load(std::string fileName);
	int loadBinary(std::string fileName);

public:
	int size;
	std::vector<float> trigs;
	std::string myname;
	void saveShellInTXT(std::string filepath);

	int readSTL(std::string fileName);
	MySTL();
	~MySTL();
};

