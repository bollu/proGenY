#pragma once
#include "../ObjProcessors/bulletProcessor.h"
#include "Generator.h"


class bulletDataGenerator{
private:
	unsigned long seed;
	unsigned int power;

	std::mt19937 generator;

	void _createBulletCollider(unsigned long colliderSeed);
public:
	bulletDataGenerator(unsigned int power, unsigned long seed);
	bulletData Generate();
};