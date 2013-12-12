#pragma once
#include "../ObjProcessors/gunProcessor.h"
#include "Generator.h"
#include "bulletDataGenerator.h"




class GunDataGenerator : public Generator{
public:
	enum Archetype{
		machineGun = 0,
		Rocket = 1,
	};

	GunDataGenerator(GunDataGenerator::Archetype archetype, 
		unsigned int power, unsigned long seed);

	GunData Generate();

private:
	

	unsigned long seed;
	unsigned int power;
	Archetype archetype;

	void _genRocket(GunData &data);
	void _genMachineGun(GunData &data);

	void _genBulletData(GunData &data,
			BulletDataGenerator::genData &generationData);
};