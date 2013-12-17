#pragma once
#include "gunsManager.h"


#include "../generators/GunDataGenerator.h"
#include "../factory/bulletCreator.h"
#include "../../core/componentSys/ObjectMgr.h"


gunsManager::gunsManager(eventMgr &eventManager, ObjectMgr &objectManager, viewProcess *viewProc, Object *player) : objectManager_(objectManager){

	eventManager.Register(Hash::getHash("nextGun"), this);
	eventManager.Register(Hash::getHash("prevGun"), this);
	eventManager.Register(Hash::getHash("playerFacingChanged"), this);
	eventManager.Register(Hash::getHash("firePlayerGun"), this);
	eventManager.Register(Hash::getHash("addGun"), this);

	player_ = player;
	currentGun_ = NULL;
	viewProc_ = viewProc;

};



#include "../factory/objectFactories.h"
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

		gotoNextGun_(skipAmt);
	}
	else if(eventName == prevGun && hasGun){

		iProp *skipProp =  dynamic_cast< iProp* >(eventData);
		int skipAmt = *skipProp->getVal();
		assert(skipAmt > 0);

		gotoPrevGun_(skipAmt);
	}
	else if(eventName == playerFacingChanged && hasGun){
		assert(eventData != NULL);
		Prop<util::Angle> *angleProp = dynamic_cast< Prop<util::Angle>* >(eventData);
		util::Angle *angle = angleProp->getVal();

		updateGunAngle_(*angle);
	}
	else if(eventName == firePlayerGun && hasGun){
		fireGun_();
	}
	else if(eventName == addGun){
		assert(eventData != NULL);

		Prop<GunDataGenerator> *GunDataGenProp = dynamic_cast< Prop<GunDataGenerator>* >(eventData);
		assert(GunDataGenProp != NULL);
		
		GunDataGenerator *GunDataGen = GunDataGenProp->getVal();
		GunData data = GunDataGen->Generate();
	
		ObjectFactories::GunFactoryInfo factoryInfo;
		factoryInfo.viewProc = viewProc_;
		factoryInfo.parent = player_;
		factoryInfo.gunData = data;
		factoryInfo.pos = *player_->getPrimitive<vector2>(Hash::getHash("position"));

		Object *gun = ObjectFactories::CreateGun(factoryInfo);
		addGun_(gun, true);
	}

};



void gunsManager::gotoNextGun_(int skip){

	currentGunIndex_ += 1;
	//wrap around
	if(currentGunIndex_ >= (guns_.size()) - 1){
		currentGunIndex_ = 0;
	}

	std::cout<<"gun index: "<<currentGunIndex_;
		switchGuns_(currentGun_, guns_[currentGunIndex_]);
	
};
void gunsManager::gotoPrevGun_(int skip){
	
	Object *oldGun = currentGun_;

	currentGunIndex_ -=1;
	//wrap around
	if(currentGunIndex_ < 0){
		currentGunIndex_ = guns_.size() - 1;
	}

	std::cout<<"gun index: "<<currentGunIndex_;
	switchGuns_(oldGun, guns_[currentGunIndex_]);
};





void gunsManager::addGun_(Object *gun, bool isCurrentGun){
	guns_.push_back(gun);
	objectManager_.addObject(gun);
	//HACK
	this->currentGun_ = gun;

	if(isCurrentGun == true){
		switchGuns_(currentGun_, gun);
	}
};




void gunsManager::updateGunAngle_(util::Angle &facing){
	if(currentGun_ == NULL){
		return;
	}

	vector2 bulletOffset = facing.polarProjection(3);
	vector2 gunOffset = facing.polarProjection(0.2);

	//get player's position
	vector2 *playerPos = player_->getPrimitive<vector2>(Hash::getHash("position"));

	//set gun's facing
	util::Angle *gunFacing = currentGun_->getPrimitive<util::Angle>(Hash::getHash("facing"));
	*gunFacing = facing;

	currentGun_->sendMessage<vector2>(Hash::getHash("setBulletSpawnPos"), *playerPos + bulletOffset);
	currentGun_->sendMessage<util::Angle>(Hash::getHash("setGunFacing"), facing);	

};

void gunsManager::fireGun_(){
	if(currentGun_ == NULL){
		return;
	}

	currentGun_->sendMessage(Hash::getHash("fireGun"));
};

void gunsManager::reloadGunPtrs_(){
	currentGun_ = guns_[currentGunIndex_];
}

void gunsManager::switchGuns_(Object *prevGun, Object *newGun){

	if(prevGun == newGun){
		return;
	}
	assert(newGun != NULL);

	if(prevGun != NULL){
		util::Angle *facing = prevGun->getPrimitive<util::Angle>(Hash::getHash("facing"));
		updateGunAngle_(*facing);
		
		objectManager_.deactivateObject(*prevGun);
		std::cout<<"\n"<<prevGun->getName()<<" was removed";
	};

	currentGun_ = newGun;
	objectManager_.activateObject(*newGun);

};