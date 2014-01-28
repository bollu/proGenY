
#include "HealthProcessor.h"


healthProcessor::healthProcessor(processMgr &processManager, Settings &settings, 
	EventManager &_eventManager) : ObjectProcessor("healthProcessor"){};

void healthProcessor::_onObjectAdd(Object *obj){
	HealthData *data = obj->getPrimitive<HealthData>(Hash::getHash("HealthData"));
	data->currentHP = data->maxHP;
};

void healthProcessor::_Process(Object *obj, float dt){

	HealthData *data = obj->getPrimitive<HealthData>(Hash::getHash("HealthData"));
	if(data->getHP() <= 0){
		obj->Kill();
	}
	
	int *HP = obj->getMessage<int>(Hash::getHash("DamageHP"));
	if (HP != NULL) {
		data->currentHP = std::max<int>(data->currentHP - *HP, 0);
	}
};

