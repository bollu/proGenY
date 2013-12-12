#pragma once
#include "../ObjProcessors/bulletProcessor.h"
#include "Generator.h"


class damageCollider;
class pushCollider;

class BulletDataGenerator : public Generator{

public:

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

	enum accelerationProperty{
		noAcceleration = 0,
		lowAcceleration,
		mediumAcceleration,
		highAcceleration
	};


	struct genData{
		gravityProperty gravity;
		damageProperty	damage;
		knockbackProperty knockback;

		//time when the latent acceleration should activate
		accelerationProperty latentAccel;
		float accelActivateTime;

		int numAbilities;
		int abilitySkill;

		genData(){
			this->numAbilities = 0;
			this->abilitySkill = 0;
			this->accelActivateTime = 0;
		}

	};

	BulletDataGenerator(genData data, 
				unsigned int power, unsigned long seed);
	BulletData Generate();

private:	
	unsigned long seed;
	unsigned int power;
	
	genData data;


	float _genGravity(gravityProperty &prop);
	float _genAcceleration(accelerationProperty &accel);

	damageCollider *_genDamage(damageProperty &prop);
	pushCollider *_genKnockback(knockbackProperty &prop); 

	BulletCollider* _createBulletCollider(unsigned long colliderSeed);
};