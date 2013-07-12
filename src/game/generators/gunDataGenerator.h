#pragma once
#include "../ObjProcessors/gunProcessor.h"
#include "Generator.h"


class gunDataGenerator{
private:
	unsigned long seed;
	unsigned int power;

	std::mt19937 generator;

	void _createBulletCollider(unsigned long colliderSeed);
public:
	gunDataGenerator(unsigned int power, unsigned long seed);
	gunData Generate();
};