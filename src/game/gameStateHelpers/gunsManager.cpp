#pragma once
#include "gunsManager.h"


#include "../generators/gunDataGenerator.h"
#include "../factory/gunCreator.h"
#include "../factory/bulletCreator.h"
#include "../../core/objectMgr.h"

#include "../factory/objectFactories.h"

gunsManager::gunsManager(eventMgr &eventManager, objectMgr &objectManager_, viewProcess *viewProc, Object *player) : objectManager_(objectManager_){

	eventManager.Register(Hash::getHash("nextGun"), this);
	eventManager.Register(Hash::getHash("prevGun"), this);
	eventManager.Register(Hash::getHash("playerFacingChanged"), this);
	eventManager.Register(Hash::getHash("firePlayerGun"), this);
	eventManager.Register(Hash::getHash("addGun"), this);

	player_ = player;
	currentGun_ = NULL;
	viewProc_ = viewProc;

};


void gunsManager::_gotoNextGun(int skip){

	currentGunIndex_ += 1;
	//wrap around
	if(currentGunIndex_ >= (guns_.size()) - 1){
		currentGunIndex_ = 0;
	}

	std::cout<<"gun index: "<<currentGunIndex_;
		_switchGuns(currentGun_, guns_[currentGunIndex_]);
	
};
void gunsManager::_gotoPrevGun(int skip){
	//_disableGun(currentGun_);

	Object *oldGun = currentGun_;

	currentGunIndex_ -=1;
	//wrap around
	if(currentGunIndex_ < 0){
		currentGunIndex_ = guns_.size() - 1;
	}

	std::cout<<"gun index: "<<currentGunIndex_;
	_switchGuns(oldGun, guns_[currentGunIndex_]);
};





void gunsManager::addGun(Object *gun, bool isCurrentGun){
	guns_.push_back(gun);

	if(isCurrentGun == true){
		_switchGuns(currentGun_, gun);
	}
};




void gunsManager::_updateGunAngle(util::Angle &facing){
	//assert(currentGun_ != NULL);
	
	if(currentGun_ == NULL){
		return;
	}

	vector2 *playerPos = player_->getProp<vector2>(Hash::getHash("position"));

	

	vector2 bulletOffset = facing.polarProjection(3);
	vector2 gunOffset = facing.polarProjection(0.2);

	currentGun_->sendMessage<vector2>(Hash::getHash("setBulletSpawnPos"), *playerPos + bulletOffset);
	currentGun_->sendMessage<util::Angle>(Hash::getHash("setGunFacing"), facing);
	currentGun_->setProp<util::Angle>(Hash::getHash("facing"), 
		&facing);

};

void gunsManager::_fireGun(){
	if(currentGun_ == NULL){
			return;
	}

	currentGun_->sendMessage(Hash::getHash("fireGun"));
};

void gunsManager::_reloadGunPtrs(){
	currentGun_ = guns_[currentGunIndex_];
//	currentGunData_ = currentGun_->getProp<GunData>(Hash::getHash("GunData"));
}


void gunsManager::recieveEvent(const Hash *eventName, baseProperty *eventData){
	const Hash *nextGun = Hash::getHash("nextGun");
	const Hash *prevGun = Hash::getHash("prevGun");
	const Hash *playerFacingChanged = Hash::getHash("playerFacingChanged");
	const Hash *firePlayerGun = Hash::getHash("firePlayerGun");
	const Hash *addGun = Hash::getHash("addGun");

	bool hasGun = (currentGun_ != NULL);
	if(eventName == nextGun && hasGun){
		iProp *skipProp =  dynamic_cast< iProp* >(eventData);
		int skipAmt = *skipProp->getVal();
		assert(skipAmt > 0);

		_gotoNextGun(skipAmt);
	}
	else if(eventName == prevGun && hasGun){

		iProp *skipProp =  dynamic_cast< iProp* >(eventData);
		int skipAmt = *skipProp->getVal();
		assert(skipAmt > 0);

		_gotoPrevGun(skipAmt);
	}
	else if(eventName == playerFacingChanged && hasGun){
		assert(eventData != NULL);
		Prop<util::Angle> *angleProp = dynamic_cast< Prop<util::Angle>* >(eventData);
		util::Angle *angle = angleProp->getVal();

		_updateGunAngle(*angle);
	}
	else if(eventName == firePlayerGun && hasGun){
		_fireGun();
	}
	else if(eventName == addGun){
		assert(eventData != NULL);

		Prop<GunDataGenerator> *GunDataGenProp = dynamic_cast< Prop<GunDataGenerator>* >(eventData);
		assert(GunDataGenProp != NULL);
		
		GunDataGenerator *GunDataGen = GunDataGenProp->getVal();
		

		GunData gunData = GunDataGen->Generate();
		//data.setBulletCreator(_bulletCreator);

		//_gunCreator->setGunData(data);
		//_gunCreator->setParent(player_);

		ObjectFactories::GunFactoryInfo factoryInfo;
		factoryInfo.viewProc = viewProc_;
		factoryInfo.parent = player_;
	factoryInfo.gunData = gunData;
		factoryInfo.pos = *player_->getProp<vector2>(Hash::getHash("position"));

		Object *gun = ObjectFactories::CreateGun(factoryInfo);
		//Object *gun = _gunCreator->createObject();
		this->addGun(gun, true);
	}

};


void gunsManager::_switchGuns(Object *prevGun, Object *newGun){

	if(prevGun == newGun){
		return;
	}
	assert(newGun != NULL);

	if(prevGun != NULL){
		util::Angle *angle = prevGun->getProp<util::Angle>(Hash::getHash("facing"));
		vector2 *pos = prevGun->getProp<vector2>(Hash::getHash("position"));


		
		newGun->setProp<util::Angle>(Hash::getHash("facing"), *angle);
		newGun->setProp<vector2>(Hash::getHash("position"), *pos);
		
		objectManager_.removeObject(prevGun);
		std::cout<<"\n"<<prevGun->getName()<<" was removed";
	};

	currentGun_ = newGun;
	//currentGunData_ = newGun->getProp<GunData>(Hash::getHash("GunData"));	
	

	objectManager_.addObject(newGun);

};
