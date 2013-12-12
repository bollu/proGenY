#pragma once
#include "gunProcessor.h"
#include "../../core/objectMgr.h"


gunProcessor::gunProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager){
		this->objectManager = processManager.getProcess<objectMgr>(Hash::getHash("objectMgrProcess"));
		this->viewProc = processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));
};


void gunProcessor::Process(float dt){
	for(cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
		Object *obj = it->second;

		GunData *gunData = obj->getProp<GunData>(Hash::getHash("GunData"));

		if(gunData == NULL){
			continue;
		}

		


		vector2 *spawnPos = obj->getMessage<vector2>(Hash::getHash("setBulletSpawnPos"));
		if(spawnPos != NULL){
			gunData->bulletPos = *spawnPos;
		}

		util::Angle *facing = obj->getMessage<util::Angle>(Hash::getHash("setGunFacing"));
		if(facing != NULL){
			gunData->facing = *facing;
		}

		gunData->_Tick(dt);
		if(obj->getMessage(Hash::getHash("fireGun")) != NULL && gunData->_canFire()){
			this->_fireShot(gunData, gunData->bulletPos);
			gunData->_Cooldown();
		}

	};
};	

#include "../factory/objectFactories.h"
void gunProcessor::_fireShot(GunData *gunData, vector2 pos){
	ObjectFactories::BulletFactoryInfo factoryInfo;

	factoryInfo.viewProc = this->viewProc;
	factoryInfo.pos = pos;
	factoryInfo.radius = gunData->bulletRadius;

	//copy bulletData from gunData onto the bullet
	factoryInfo.bulletData = gunData->bulletData;
	factoryInfo.bulletData.beginVel = gunData->facing.toVector() * gunData->bulletVel;
	factoryInfo.bulletData.angle = util::Angle(gunData->facing);

	Object *bullet = ObjectFactories::CreateBullet(factoryInfo);
	this->objectManager->addObject(bullet);

};

void gunProcessor::postProcess(){};


void GunData::_Tick(float dt){

	if(this->clipCooldown.onCooldown()) {
		
		//cooldown transitioned from on to off	
		if(this->clipCooldown.Tick(dt).offCooldown()) {
			this->currentClipSize = this->totalClipSize;
		}
	}

	if(this->shotCooldown.onCooldown()) {
		this->shotCooldown.Tick(dt);
	}
};


void GunData::_Cooldown(){
	this->currentClipSize--;

	if (this->currentClipSize <= 0) {
		this->clipCooldown.startCooldown();
	} else { 
		this->shotCooldown.startCooldown();
	}
};

bool GunData::_canFire(){
	return (this->clipCooldown.offCooldown() && this->shotCooldown.offCooldown());
}