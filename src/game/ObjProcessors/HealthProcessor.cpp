#pragma once
#include "HealthProcessor.h"


healthProcessor::healthProcessor(processMgr &processManager, Settings &settings, 
	eventMgr &_eventManager) : ObjectProcessor("healthProcessor"){};


void healthProcessor::_Process(Object *obj, float dt){

	healthData *data = obj->getPrimitive<healthData>(Hash::getHash("healthData"));
	if(data->getHP() <= 0){
		obj->Kill();
	}
};

