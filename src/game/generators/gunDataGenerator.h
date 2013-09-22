#pragma once
#include "../ObjProcessors/gunProcessor.h"
#include "Generator.h"
#include "bulletDataGenerator.h"




class gunDataGenerator : public Generator{
public:
	enum Archetype{
		machineGun = 0,
		Rocket = 1,
	};

	gunDataGenerator(gunDataGenerator::Archetype archetype, 
		unsigned int power, unsigned long seed);

	gunData Generate();



private:

	

	unsigned long seed;
	unsigned int power;
	Archetype archetype;

	void _genRocket(gunData &data);
	void _genMachineGun(gunData &data);

	void _genBulletData(gunData &data,
			bulletDataGenerator::genData &generationData);
};
