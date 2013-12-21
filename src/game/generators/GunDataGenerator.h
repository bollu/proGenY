#pragma once
#include "../ObjProcessors/GunProcessor.h"
#include "BulletDataGenerator.h"

enum GunType{
		machineGun = 0,
		Rocket = 1,
};

struct GunGenData {
	unsigned int power;
	unsigned long seed;

	GunType type;
};

GunData GenGunData(GunGenData &genData);
