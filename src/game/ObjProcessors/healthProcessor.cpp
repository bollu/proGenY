#pragma once
#include "healthProcessor.h"


healthProcessor::healthProcessor(processMgr &processManager, Settings &settings, 
	eventMgr &_eventManager) : objectProcessor("healthProcessor"){};


void healthProcessor::_Process(Object *obj, float dt){

	healthData *data = obj->getPrimitive<healthData>(Hash::getHash("healthData"));
	if(data->getHP() <= 0){
		obj->Kill();
	}
};

