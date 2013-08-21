#pragma once
#include "../ObjProcessors/bulletProcessor.h"
#include "Generator.h"


class damageCollider;
class pushCollider;
class bulletPropGenerator : public Generator
{
public:
	enum gravityProperty
	{
		noGravity = 0, lowGravity, defaultGravity, highGravity, };


	enum damageProperty
	{
		lowDamage = 0, mediumDamage, highDamage, };


	enum knockbackProperty
	{
		noKnockback = 0, lowKnockback, mediumKnockback, highKnockback
	};


	struct genData {
		gravityProperty	  gravity;
		damageProperty	  damage;
		knockbackProperty knockback;
		int		  numAbilities;
		int		  abilitySkill;
		genData (){
			this->numAbilities = 0;
			this->abilitySkill = 0;
		}
	};


	bulletPropGenerator ( genData data, unsigned int power, unsigned long seed );

	bulletProp *Generate();


private:
	unsigned long seed;
	unsigned int power;
	genData data;

	float _genGravity ( gravityProperty &prop );

	damageCollider	*_genDamage ( damageProperty &prop );
	pushCollider	*_genKnockback ( knockbackProperty &prop );
	bulletCollider * _createBulletCollider ( unsigned long colliderSeed );
};