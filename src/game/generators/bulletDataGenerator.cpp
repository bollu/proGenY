#pragma once
#include "bulletDataGenerator.h"

BulletDataGenerator::BulletDataGenerator(genData data,
	unsigned int power, unsigned long seed){
	
	this->seed = seed;
	this->power = power;
	this->data = data;

	generator.seed(seed);
};



#pragma once
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../ObjProcessors/bulletProcessor.h"
#include "../bulletColliders/damageCollider.h"
#include "../bulletColliders/pushCollider.h"




BulletData BulletDataGenerator::Generate(){
	BulletData bullet;

	bullet.gravityScale = this->_genGravity(this->data.gravity);

	bullet.addLatentAcceleration(data.accelActivateTime, 
			_genAcceleration(this->data.latentAccel));


	pushCollider *push = this->_genKnockback(this->data.knockback);
	damageCollider *damage = this->_genDamage(this->data.damage);


	if(push != NULL){
		bullet.addBulletCollder(push);
	}
	if(damage != NULL){
		bullet.addBulletCollder(damage);
	}

	for(int i = 0; i < this->data.numAbilities; i++){
		
		bullet.addBulletCollder(this->_createBulletCollider(this->seed));
	}	
	return bullet;
};

float BulletDataGenerator::_genGravity(gravityProperty &prop){
	switch(prop){
		case noGravity:
			return 0;

		case lowGravity:
			return this->_genFloat(0.4, 0.8);

		case defaultGravity:
			return 1.0;

		case highGravity:
			return this->_genFloat(2.0, 4.0); 
	};
};

float BulletDataGenerator::_genAcceleration(accelerationProperty &accel){
	switch(accel){
		case noAcceleration:
			return 0;

		case lowAcceleration:
			return this->_genFloat(0.4, 0.8);

		case mediumAcceleration:
			return 5.0;

		case highAcceleration:
			return this->_genFloat(40.0, 50.2); 
	};
};


#include "../bulletColliders/damageCollider.h"
damageCollider *BulletDataGenerator::_genDamage(damageProperty &prop){
	float damage;

	switch(prop){
		case lowDamage:
		damage = this->_genFloat(1.0, 5.0);
		break;

		case mediumDamage:
		damage = this->_genFloat(10.0, 20.0);
		break;

		case highDamage:
		damage = this->_genFloat(25.0, 30.0);
		break;
	};

	return new damageCollider(damage);
	//return new damageCollider(0);
};

#include "../bulletColliders/pushCollider.h"
pushCollider *BulletDataGenerator::_genKnockback(knockbackProperty &prop){
	if(prop == noKnockback){
		return NULL;
	}

	float knockback;
	switch(prop){
		case lowKnockback:
			knockback = this->_genFloat(1.0, 4.0);
			break;

		case mediumKnockback:
			knockback = this->_genFloat(8.0, 13.0);
			break;

		case highKnockback:
			knockback = this->_genFloat(15.0, 20.0);
			break;
	}

	

	return new pushCollider(knockback);
};

#include "../bulletColliders/bounceCollider.h"
BulletCollider* BulletDataGenerator::_createBulletCollider(unsigned long colliderSeed){

	static const int numAbilities = 1;
	int type = this->_genInt(0.0, numAbilities - 1);

	switch (type){
		case 0:
			return new bounceCollider(1);
		break;

		default:
			util::errorLog<<"ability index out of bounds";

	};

};


