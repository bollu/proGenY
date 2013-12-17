	#pragma once
#include "GunDataGenerator.h"
#include "BulletDataGenerator.h"


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

	/*BulletDataGenerator::gravityProperty gForce;

	switch(archetype){
		case Archetype::Rocket:
			gForce = BulletDataGenerator::gravityProperty::highGravity;
			break;

		case Archetype::machineGun:
			gForce = BulletDataGenerator::gravityProperty::defaultGravity;
			break;
	}*/

	BulletDataGenerator BulletDataGen(generationData, this->power, this->seed);
	BulletData _BulletData = BulletDataGen.Generate();


	_BulletData.addEnemyCollision(Hash::getHash("enemy"));
	_BulletData.addEnemyCollision(Hash::getHash("dummy"));

	_BulletData.addIgnoreCollision(Hash::getHash("bullet"));
	//_BulletData.addIgnoreCollision(Hash::getHash("terrain"));
	//_BulletData.addIgnoreCollision(Hash::getHash("boundary"));
	_BulletData.addIgnoreCollision(Hash::getHash("player"));

	data.setBulletData(_BulletData);

};


void GunDataGenerator::_genRocket(GunData &data){
	/* quite a low rate of fire, average
	clip size, heavy damage
	*/

	//clip size can vary from power-6  to powe6   
	data.setClipSize(1); //this->_normGenInt(6 + power, 1));
	data.setShotCooldown(0);


	int clipCD = this->_normGenInt(16 + power, 2);
	data.setClipCooldown(clipCD);

	data.setBulletRadius(this->_genFloat(0.8, 0.9));
	data.setBulletVel(this->_genFloat(40, 60));

	BulletDataGenerator::genData bulletGenData
					(BulletDataGenerator::knockbackProperty::highKnockback,
					BulletDataGenerator::gravityProperty::highGravity,
					BulletDataGenerator::damageProperty::highDamage,
					1,
					1);


	this->_genBulletData(data, bulletGenData);

};


void GunDataGenerator::_genMachineGun(GunData &data){

	/*high rate of fire, small bullets, medium clip size */

	data.setClipSize( this->_normGenInt(20, 2));

	int shotCooldown = this->_normGenInt(3, 1);
	data.setShotCooldown(shotCooldown);
	data.setClipCooldown(this->_normGenInt(shotCooldown + 20, 2));


	data.setBulletRadius(this->_genFloat(0.4, 0.5));

	data.setBulletVel(this->_genFloat(20, 30));

		BulletDataGenerator::genData bulletGenData
					(BulletDataGenerator::knockbackProperty::noKnockback,
					BulletDataGenerator::gravityProperty::lowGravity,
					BulletDataGenerator::damageProperty::lowDamage,
					1,
					1);

	this->_genBulletData(data, bulletGenData);
};
