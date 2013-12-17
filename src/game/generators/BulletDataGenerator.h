#pragma once
#include "../ObjProcessors/BulletProcessor.h"
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



	struct genData{
		gravityProperty gravity;
		damageProperty	damage;
		knockbackProperty knockback;

		int numAbilities;
		int abilitySkill;

		genData(knockbackProperty knockback,
				gravityProperty gravity,
				damageProperty damage,
				unsigned int numAbilities,
				unsigned int abilitySkill){
			
			this->numAbilities = numAbilities;
			this->abilitySkill = abilitySkill;
			this->knockback = knockback;
			this->damage = damage;
			this->gravity = gravity;
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
	damageCollider *_genDamage(damageProperty &prop);
	pushCollider *_genKnockback(knockbackProperty &prop); 

	BulletCollider* _createBulletCollider(unsigned long colliderSeed);
};
