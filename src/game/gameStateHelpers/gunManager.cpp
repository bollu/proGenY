#pragma once
#include "gunsManager.h"


#include "../generators/gunDataGenerator.h"
#include "../factory/gunCreator.h"
#include "../factory/bulletCreator.h"
#include "../../core/objectMgr.h"


gunsManager::gunsManager(eventMgr &eventManager, objectFactory &_factory, 
		objectMgr &_objectManager, Object *player) : objectManager(_objectManager){

	eventManager.Register(Hash::getHash("nextGun"), this);
	eventManager.Register(Hash::getHash("prevGun"), this);
	eventManager.Register(Hash::getHash("playerFacingChanged"), this);
	eventManager.Register(Hash::getHash("firePlayerGun"), this);
	eventManager.Register(Hash::getHash("addGun"), this);


	this->_gunCreator = (gunCreator *) _factory.getCreator(
						Hash::getHash("gun"));

	this->_bulletCreator = (bulletCreator *) _factory.getCreator(
						Hash::getHash("bullet"));

	this->player = player;
	this->currentGun = NULL;

};


void gunsManager::_gotoNextGun(){

	//wrap around
	if(this->currentGunIndex == this->guns.size() - 1){
		this->currentGunIndex = 0;
	}else{
		this->currentGunIndex++;
	}

	this->_reloadGunPtrs();

	
};
void gunsManager::_gotoPrevGun(){
	//wrap around
	if(this->currentGunIndex == 0){
		this->currentGunIndex = this->guns.size() - 1;
	}else{
		this->currentGunIndex--;
	}

	this->_reloadGunPtrs();
};
	




void gunsManager::addGun(Object *gun, bool isCurrentGun){
	this->guns.push_back(gun);

	if(isCurrentGun == true){
		this->currentGun = gun;
		this->currentGunIndex = this->guns.size() - 1;
		
		this->_reloadGunPtrs();
	}
};




void gunsManager::_updateGunAngle(util::Angle &facing){
	assert(this->currentGun != NULL);
	vector2 *playerPos = this->player->getProp<vector2>(Hash::getHash("position"));

	this->currentGun->setProp<util::Angle>(Hash::getHash("facing"), 
		&facing);
				
	vector2 bulletOffset = facing.polarProjection(3);
	vector2 gunOffset = facing.polarProjection(0.2);


	this->currentGunData->setBulletPos(*playerPos + bulletOffset);
	this->currentGunData->setFacing(facing);


};

void gunsManager::_fireGun(){
	assert(this->currentGun != NULL);

	this->currentGunData->Fire();
};

void gunsManager::_reloadGunPtrs(){
	this->currentGun = this->guns[this->currentGunIndex];
	this->currentGunData = this->currentGun->getProp<gunData>(Hash::getHash("gunData"));
}


void gunsManager::recieveEvent(const Hash *eventName, baseProperty *eventData){
	const Hash *nextGun = Hash::getHash("nextGun");
	const Hash *prevGun = Hash::getHash("prevGun");
	const Hash *playerFacingChanged = Hash::getHash("playerFacingChanged");
	const Hash *firePlayerGun = Hash::getHash("firePlayerGun");
	const Hash *addGun = Hash::getHash("addGun");
	

	if(eventName == nextGun){
		this->_gotoNextGun();
	}
	else if(eventName == prevGun){
		this->_gotoPrevGun();
	}
	else if(eventName == playerFacingChanged){
		assert(eventData != NULL);
		Prop<util::Angle> *angleProp = dynamic_cast< Prop<util::Angle>* >(eventData);
		util::Angle *angle = angleProp->getVal();

		
		this->_updateGunAngle(*angle);
	}
	else if(eventName == firePlayerGun){
		this->_fireGun();
	}

	else if(eventName == addGun){
		assert(eventData != NULL);

		Prop<gunDataGenerator> *gunDataGenProp = dynamic_cast< Prop<gunDataGenerator>* >(eventData);
		assert(gunDataGenProp != NULL);
		gunDataGenerator *gunDataGen = gunDataGenProp->getVal();
		

		gunData data = (*gunDataGen).Generate();
		data.setBulletCreator(this->_bulletCreator);

		this->_gunCreator->setGunData(data);
		this->_gunCreator->setParent(player);

		Object *gun = _gunCreator->createObject(*player->getProp<vector2>(Hash::getHash("position")));
		this->addGun(gun, true);


		this->objectManager.addObject(gun);


		//util::msgLog("got the gunData");
	}

};