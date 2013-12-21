#pragma once
#include "../ObjProcessors/BulletProcessor.h"



class damageCollider;
class pushCollider;

enum gravityProperty{
		noGravity = 0,
		lowGravity,
		defaultGravity,
		highGravity,
};

enum damageProperty{
	lowDamage = 0,
	mediumDamage,
	highDamage,
};

enum knockbackProperty{
	noKnockback = 0,
	lowKnockback,
	mediumKnockback,
	highKnockback
};

struct BulletGenData {
	unsigned long seed;
	unsigned  int power;

	unsigned int numAbilities;
	unsigned int abilityPower;
	
	gravityProperty gravity;
	damageProperty damage;
	knockbackProperty knockback;
};

BulletData GenBulletData(BulletGenData &data);