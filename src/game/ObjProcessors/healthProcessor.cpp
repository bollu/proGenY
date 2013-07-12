#pragma once
#include "healthProcessor.h"



void healthProcessor::postProcess(){
	for(auto it=  objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;

		healthData *data = obj->getProp<healthData>(Hash::getHash("healthData"));
		if(data == NULL){
			continue;

			
		}	

		if(data->getHP() <= 0){
			obj->Kill();
		}
	};
};

