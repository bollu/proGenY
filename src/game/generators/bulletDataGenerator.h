#pragma once
#include "../ObjProcessors/bulletProcessor.h"
#include "Generator.h"


class damageCollider;
class pushCollider;

class bulletDataGenerator : public Generator{

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

	bulletDataGenerator(genData data, 
				unsigned int power, unsigned long seed);
	bulletData Generate();

private:	
	unsigned long seed;
	unsigned int power;
	
	genData data;


	float _genGravity(gravityProperty &prop);
	damageCollider *_genDamage(damageProperty &prop);
	pushCollider *_genKnockback(knockbackProperty &prop); 

	bulletCollider* _createBulletCollider(unsigned long colliderSeed);
};
