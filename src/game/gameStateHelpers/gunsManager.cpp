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


this->_gunCreator =  _factory.getCreator<gunCreator>(
		Hash::getHash("gun"));

	this->_bulletCreator = _factory.getCreator<bulletCreator>(
		Hash::getHash("bullet"));

	this->player = player;
	this->currentGun = NULL;

};


void gunsManager::_gotoNextGun(int skip){

	this->currentGunIndex += 1;
	//wrap around
	if(this->currentGunIndex >= (this->guns.size()) - 1){
		this->currentGunIndex = 0;
	}

	std::cout<<"gun index: "<<this->currentGunIndex;
		this->_switchGuns(this->currentGun, this->guns[this->currentGunIndex]);
	
};
void gunsManager::_gotoPrevGun(int skip){
	//this->_disableGun(this->currentGun);

	Object *oldGun = this->currentGun;

	this->currentGunIndex -=1;
	//wrap around
	if(this->currentGunIndex < 0){
		this->currentGunIndex = this->guns.size() - 1;
	}

	std::cout<<"gun index: "<<this->currentGunIndex;
	this->_switchGuns(oldGun, this->guns[this->currentGunIndex]);
};





void gunsManager::addGun(Object *gun, bool isCurrentGun){
	this->guns.push_back(gun);

	if(isCurrentGun == true){
		
		this->_switchGuns(this->currentGun, gun);
	}
};




void gunsManager::_updateGunAngle(util::Angle &facing){
	//assert(this->currentGun != NULL);
	
	if(this->currentGun == NULL){
		return;
	}

	vector2 *playerPos = this->player->getProp<vector2>(Hash::getHash("position"));

	util::Angle *gunAngle = this->currentGun->getProp<util::Angle>(Hash::getHash("facing"));
	*gunAngle = facing;

	vector2 bulletOffset = facing.polarProjection(3);
	vector2 gunOffset = facing.polarProjection(0.2);


	this->currentGunData->setBulletPos(*playerPos + bulletOffset);
	this->currentGunData->setFacing(facing);


};

void gunsManager::_fireGun(){
	if(this->currentGun == NULL){
			return;
	}

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

	bool hasGun = (this->currentGun != NULL);
	if(eventName == nextGun && hasGun){
		iProp *skipProp =  dynamic_cast< iProp* >(eventData);
		int skipAmt = *skipProp->getVal();
		assert(skipAmt > 0);

		this->_gotoNextGun(skipAmt);
	}
	else if(eventName == prevGun && hasGun){

		iProp *skipProp =  dynamic_cast< iProp* >(eventData);
		int skipAmt = *skipProp->getVal();
		assert(skipAmt > 0);

		this->_gotoPrevGun(skipAmt);
	}
	else if(eventName == playerFacingChanged && hasGun){
		assert(eventData != NULL);
		Prop<util::Angle> *angleProp = dynamic_cast< Prop<util::Angle>* >(eventData);
		util::Angle *angle = angleProp->getVal();

		this->_updateGunAngle(*angle);
	}
	else if(eventName == firePlayerGun && hasGun){
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
	}

};


void gunsManager::_switchGuns(Object *prevGun, Object *newGun){

	if(prevGun == newGun){
		return;
	}
	assert(newGun != NULL);

	if(prevGun != NULL){
		util::Angle *prevAngle = prevGun->getProp<util::Angle>(Hash::getHash("facing"));
		vector2 *prevPos = prevGun->getProp<vector2>(Hash::getHash("position"));


		util::Angle *newAngle = newGun->getProp<util::Angle>(Hash::getHash("facing"));
		vector2 *newPos = newGun->getProp<vector2>(Hash::getHash("position"));
		
		*newAngle = *prevAngle;
		*newPos = *prevPos;
		
		this->objectManager.removeObject(prevGun);
		std::cout<<"\n"<<prevGun->getName()<<" was removed";
	};

	this->currentGun = newGun;
	this->currentGunData = newGun->getProp<gunData>(Hash::getHash("gunData"));	
	

	this->objectManager.addObject(newGun);

};
