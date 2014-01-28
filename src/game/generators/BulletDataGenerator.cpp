#include "BulletDataGenerator.h"
#include "Generator.h"

#include "../../core/componentSys/processor/PhyProcessor.h"
#include "../ObjProcessors/BulletProcessor.h"
#include "../BulletColliders/damageCollider.h"
#include "../BulletColliders/PushCollider.h"

float _genGravity(std::mt19937 &generator, gravityProperty &prop){
	switch(prop){
		case noGravity:
			return 0;

		case lowGravity:
			return genFloat(generator, 0.4, 0.8);

		case defaultGravity:
			return 1.0;

		case highGravity:
			return genFloat(generator, 2.0, 4.0); 
	};
};



#include "../BulletColliders/damageCollider.h"
damageCollider *_genDamage(std::mt19937 &generator, damageProperty &prop){
	float damage;

	switch(prop){
		case lowDamage:
		damage = genFloat(generator, 1.0, 5.0);
		break;

		case mediumDamage:
		damage = genFloat(generator, 10.0, 20.0);
		break;

		case highDamage:
		damage = genFloat(generator, 25.0, 30.0);
		break;
	};

	return new damageCollider(damage);
};

#include "../BulletColliders/PushCollider.h"
pushCollider *_genKnockback(std::mt19937 &generator, knockbackProperty &prop){
	if(prop == noKnockback){
		return NULL;
	}

	float knockback;
	switch(prop){
		case noKnockback:
			knockback = 0;
			break;

		case lowKnockback:
			knockback = genFloat(generator, 1.0, 4.0);
			break;

		case mediumKnockback:
			knockback = genFloat(generator, 8.0, 13.0);
			break;

		case highKnockback:
			knockback = genFloat(generator, 15.0, 20.0);
			break;
	default:
		IO::errorLog<<"passing an unknown knockback enumeration"<<IO::flush;

	}

	return new pushCollider(knockback);
};




#include "../BulletColliders/bounceCollider.h"
StabCollider* _createAbility(std::mt19937 &generator, unsigned int abilityPower){
	//HACK!!!
	return new bounceCollider(abilityPower);


	static const int totalAbilities = 1;
	int type = genInt(generator, 0.0, totalAbilities - 1);

	switch (type){
		case 0:
			return new bounceCollider(abilityPower);
			break;

		default:
			IO::errorLog<<"ability index out of bounds"<<IO::flush;;
			return NULL;
	};

};


StabData GenStabData(BulletGenData &data) {
	static std::mt19937 generator;
	generator.seed(data.seed);
	
	StabData stabData;

	pushCollider *push =  _genKnockback(generator, data.knockback);
	damageCollider *damage = _genDamage(generator, data.damage);

	if(push != NULL){
		stabData.addCollider(push);
	}
	if(damage != NULL){
		stabData.addCollider(damage);
	}

	//HACK!
	/*
	for(int i = 0; i < data.numAbilities; i++){
		stabData.addCollider(_createAbility(generator, data.abilityPower));
	}*/

	stabData.addCollider(new bounceCollider(2));
	return stabData;
}

BulletData GenBulletData(BulletGenData &data){
	static std::mt19937 generator;
	generator.seed(data.seed);
	
	BulletData bulletData;
	bulletData.gravityScale = _genGravity(generator, data.gravity);
	
	/*
	pushCollider *push =  _genKnockback(generator, data.knockback);
	damageCollider *damage = _genDamage(generator, data.damage);


	if(push != NULL){
		bulletData.addBulletCollder(push);
	}
	if(damage != NULL){
		bulletData.addBulletCollder(damage);
	}

	for(int i = 0; i < data.numAbilities; i++){
		bulletData.addBulletCollder(_createBulletCollider(generator, data.abilityPower));
	}	
	*/
	return bulletData;
};
