#pragma once
#include "gunProcessor.h"
#include "../../core/Process/objectMgrProcess.h"


gunProcessor::gunProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager){
		objectMgrProcess *objMgrProc = processManager.getProcess<objectMgrProcess>(
			Hash::getHash("objectMgrProcess"));
		this->objectManager = objMgrProc->getObjectMgr();
};


void gunProcessor::Process(float dt){
	for(cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
		Object *obj = it->second;

		gunData *data = obj->getProp<gunData>(Hash::getHash("gunData"));

		if(data == NULL){
			continue;
		}

		data->_Tick();

		if(data->_shouldFire()){
			this->_fireShot(data, data->bulletPos);

			data->_Cooldown();
		}
	};
};	

void gunProcessor::_fireShot(gunData *data, vector2 pos){
	bulletCreator *creator = data->creator;
	assert(creator != NULL);

	vector2 beginVel = data->facing.toVector() * data->buletVel;

	data->bullet.beginVel = beginVel;

	creator->setBulletData(data->bullet);
	creator->setCollisionRadius(data->bulletRadius);
	Object *bullet = creator->createObject(pos);
	
	this->objectManager->addObject(bullet);
};

void gunProcessor::postProcess(){};


void gunData::_Tick(){
	if(this->clipOnCooldown){	

		if(this->currentClipCooldown == 0){
			this->clipOnCooldown = false;
			this->currentClipSize = this->totalClipSize;
		}

		this->currentClipCooldown--;

	}

	if(this->shotOnCooldown){
		if(this->currrentShotCooldown == 0){
			this->shotOnCooldown = false;
		}

		this->currrentShotCooldown--;
	}
};

void gunData::Fire(){

	if(!this->clipOnCooldown && !this->shotOnCooldown){
		this->firing = true;
	};
};

void gunData::_Cooldown(){
	this->firing = false;

	this->currentClipSize--;

	if(this->currentClipSize <= 0){
		this->clipOnCooldown = true;
		this->currentClipCooldown = this->totalClipCooldown;
	}else{
		this->shotOnCooldown = true;
		this->currrentShotCooldown = this->totalShotCooldown;
	}
};