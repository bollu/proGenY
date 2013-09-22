#pragma once
#include "gunProcessor.h"
#include "../../core/Process/objectMgrProcess.h"


gunProcessor::gunProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager) 
: objectProcessor("gunProcessor"){
	objectMgrProcess *objMgrProc = processManager.getProcess<objectMgrProcess>(
		Hash::getHash("objectMgrProcess"));
	this->objectManager = objMgrProc->getObjectMgr();
};


void gunProcessor::_Process(Object *obj, float dt){
	
	gunData *data = obj->getPrimitive<gunData>(Hash::getHash("gunData"));

	data->_Tick();

	if(data->_shouldFire()){
		this->_fireShot(data, data->bulletPos);

		data->_Cooldown();
	}
};	

void gunProcessor::_fireShot(gunData *data, vector2 pos){
	bulletCreator *creator = data->creator;
	assert(creator != NULL);

	vector2 beginVel = data->facing.toVector() * data->buletVel;
	data->bullet.beginVel = beginVel;
	
	creator->Init(data->bullet, data->bulletRadius);

	Object *bullet = creator->createObject(pos);	
	this->objectManager->addObject(bullet);
};

void gunProcessor::_onObjectDeactivate(Object *obj){

	gunData *data = obj->getPrimitive<gunData>(Hash::getHash("gunData"));
	//stop the gun from firing if it's deactivated. 
	data->firing = false;

};


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
