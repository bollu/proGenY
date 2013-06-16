#pragma once
#include "gunProcessor.h"

void gunData::_fireShot(){};

void gunData::Tick(){};

void gunProcessor::Process(float dt){
	for(cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
		Object *obj = it->second;

		gunData *data = obj->getProp<gunData>(Hash::getHash("gunData"));

		if(data == NULL){
			continue;
		}

		data->Tick();

		if(data->shouldFire()){
			data->_fireShot();
		}
	};
};
void gunProcessor::postProcess(){};
