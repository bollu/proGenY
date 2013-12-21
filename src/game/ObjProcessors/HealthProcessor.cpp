#pragma once
#include "HealthProcessor.h"


healthProcessor::healthProcessor(processMgr &processManager, Settings &settings, 
	EventManager &_eventManager) : ObjectProcessor("healthProcessor"){};

void healthProcessor::_onObjectAdd(Object *obj){
	healthData *data = obj->getPrimitive<healthData>(Hash::getHash("healthData"));
	data->currentHP = data->maxHP;
};

void healthProcessor::_Process(Object *obj, float dt){

	healthData *data = obj->getPrimitive<healthData>(Hash::getHash("healthData"));
	if(data->getHP() <= 0){
		obj->Kill();
	}
	
	int *HP = obj->getMessage<int>(Hash::getHash("DamageHP"));
	if (HP != NULL) {
		data->currentHP = std::max<int>(data->currentHP - *HP, 0);
	}
};

