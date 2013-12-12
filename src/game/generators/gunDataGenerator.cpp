#pragma once
#include "gunDataGenerator.h"
#include "bulletDataGenerator.h"


GunDataGenerator::GunDataGenerator(GunDataGenerator::Archetype archetype, 
	unsigned int power, unsigned long seed){
	
	this->power = power;
	this->seed = seed;
	this->archetype = archetype;

	assert(this->power > 0);

	//HACK
	generator.seed(seed + rand());
};


GunData GunDataGenerator::Generate(){

	
	GunData data;

	switch(this->archetype){
		case Archetype::Rocket:
			this->_genRocket(data);
			break;
		
		case Archetype::machineGun:
			this->_genMachineGun(data);
			break;
	}


	return data;	
};

void GunDataGenerator::_genBulletData(GunData &data,
	BulletDataGenerator::genData &generationData){

	BulletDataGenerator BulletDataGen(generationData, this->power, this->seed);
	BulletData _BulletData = BulletDataGen.Generate();


	_BulletData.addEnemyCollision(Hash::getHash("enemy"));
	_BulletData.addEnemyCollision(Hash::getHash("dummy"));

	/*
	_BulletData.addIgnoreCollision(Hash::getHash("bullet"));
	//_BulletData.addIgnoreCollision(Hash::getHash("terrain"));
	//_BulletData.addIgnoreCollision(Hash::getHash("boundary"));
	_BulletData.addIgnoreCollision(Hash::getHash("player"));*/

	data.setBulletData(_BulletData);

};


void GunDataGenerator::_genRocket(GunData &data){
	/* quite a low rate of fire, average
	clip size, heavy damage
	*/

	//clip size can vary from power-6  to powe6   
	data.setClipSize(1); //this->_normGenInt(6 + power, 1));
	data.setShotCooldown(0);


	float clipCD = this->_normGenInt(500 + power * 10, 10) / 1000.0;
	data.setClipCooldown(clipCD);

	

	data.setBulletRadius(this->_genFloat(0.8, 0.9));
	
	//data.setBulletRadius(this->_genFloat(0.2, 0.4));
	data.setBulletVel(this->_genFloat(5, 10));

	BulletDataGenerator::genData bulletGenData;
	
	bulletGenData.gravity = BulletDataGenerator::gravityProperty::noGravity;

	bulletGenData.latentAccel = BulletDataGenerator::accelerationProperty::highAcceleration;
	bulletGenData.accelActivateTime = 0.2;

	bulletGenData.knockback = BulletDataGenerator::knockbackProperty::highKnockback;
	bulletGenData.damage = BulletDataGenerator::damageProperty::highDamage;
	bulletGenData.numAbilities = 0;

	this->_genBulletData(data, bulletGenData);

};


void GunDataGenerator::_genMachineGun(GunData &data){

	/*high rate of fire, small bullets, medium clip size */

	data.setClipSize( this->_normGenInt(20, 2));

	//cooldowns are in seconds
	float shotCooldown = this->_normGenFloat(30, 10) / 1000.0;
	data.setShotCooldown(shotCooldown);
	data.setClipCooldown(this->_normGenFloat(shotCooldown + 20 / 1000.0, 2 / 1000.0));


	data.setBulletRadius(this->_genFloat(0.4, 0.5));

	data.setBulletVel(this->_genFloat(20, 30));

	BulletDataGenerator::genData bulletGenData;
	bulletGenData.gravity = BulletDataGenerator::gravityProperty::lowGravity;
	bulletGenData.numAbilities = 1;
	bulletGenData.numAbilities = 1;

	this->_genBulletData(data, bulletGenData);
};
