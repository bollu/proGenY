#pragma once
#include "gunDataGenerator.h"
#include "bulletDataGenerator.h"


gunDataGenerator::gunDataGenerator(gunDataGenerator::Archetype archetype, 
	unsigned int power, unsigned long seed){
	
	this->power = power;
	this->seed = seed;
	this->archetype = archetype;

	assert(this->power > 0);

	//HACK
	generator.seed(seed + rand());
};


gunData gunDataGenerator::Generate(){

	
	gunData data;

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

void gunDataGenerator::_genBulletData(gunData &data,
	bulletDataGenerator::genData &generationData){

	/*bulletDataGenerator::gravityProperty gForce;

	switch(archetype){
		case Archetype::Rocket:
			gForce = bulletDataGenerator::gravityProperty::highGravity;
			break;

		case Archetype::machineGun:
			gForce = bulletDataGenerator::gravityProperty::defaultGravity;
			break;
	}*/

	bulletDataGenerator bulletDataGen(generationData, this->power, this->seed);
	bulletData _bulletData = bulletDataGen.Generate();


	_bulletData.addEnemyCollision(Hash::getHash("enemy"));
	_bulletData.addEnemyCollision(Hash::getHash("dummy"));

	_bulletData.addIgnoreCollision(Hash::getHash("bullet"));
	_bulletData.addIgnoreCollision(Hash::getHash("bullet"));
	_bulletData.addIgnoreCollision(Hash::getHash("bullet"));
	//_bulletData.addIgnoreCollision(Hash::getHash("terrain"));
	//_bulletData.addIgnoreCollision(Hash::getHash("boundary"));
	_bulletData.addIgnoreCollision(Hash::getHash("player"));

	data.setBulletData(_bulletData);

};


void gunDataGenerator::_genRocket(gunData &data){
	/* quite a low rate of fire, average
	clip size, heavy damage
	*/

	//clip size can vary from power-6  to powe6   
	data.setClipSize(1); //this->_normGenInt(6 + power, 1));
	data.setShotCooldown(0);


	int clipCD = this->_normGenInt(16 + power, 2);
	data.setClipCooldown(clipCD);

	

	data.setBulletRadius(this->_genFloat(0.8, 0.9));
//	data.setBulletRadius(this->_genFloat(0.2, 0.4));
	data.setBulletVel(this->_genFloat(40, 60));

	bulletDataGenerator::genData bulletGenData;
	
	bulletGenData.gravity = bulletDataGenerator::gravityProperty::highGravity;
	bulletGenData.knockback = bulletDataGenerator::knockbackProperty::highKnockback;
	bulletGenData.damage = bulletDataGenerator::damageProperty::highDamage;
	bulletGenData.numAbilities = 1;

	this->_genBulletData(data, bulletGenData);

};


void gunDataGenerator::_genMachineGun(gunData &data){

	/*high rate of fire, small bullets, medium clip size */

	data.setClipSize( this->_normGenInt(20, 2));

	int shotCooldown = this->_normGenInt(3, 1);
	data.setShotCooldown(shotCooldown);
	data.setClipCooldown(this->_normGenInt(shotCooldown + 20, 2));


	data.setBulletRadius(this->_genFloat(0.4, 0.5));

	data.setBulletVel(this->_genFloat(20, 30));

	bulletDataGenerator::genData bulletGenData;
	bulletGenData.gravity = bulletDataGenerator::gravityProperty::lowGravity;
	bulletGenData.numAbilities = 1;
	bulletGenData.numAbilities = 1;

	this->_genBulletData(data, bulletGenData);
};
