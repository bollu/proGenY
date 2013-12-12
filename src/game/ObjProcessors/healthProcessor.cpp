#pragma once
#include "healthProcessor.h"



void healthProcessor::postProcess(){
	for(auto it=  objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;

		HealthData *data = obj->getProp<HealthData>(Hash::getHash("HealthData"));
		if(data == NULL){
			continue;		
		}

		float *damage = obj->getMessage<float>(Hash::getHash("damageHealth"));
		if(damage != NULL) {
			data->Damage(*damage);
		}

		if(data->getHP() <= 0){
			obj->Kill();
		}
	};
};

