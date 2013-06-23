#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"

#include "../../core/Process/viewProcess.h"
#include "../../core/Process/worldProcess.h"
#include "../../util/mathUtil.h"


struct offsetData{
public:
	vector2 posOffset;
	util::Angle angleOffset;
	bool centered;

	Object *parent;

};


class offsetProcessor : public objectProcessor{
public:
	offsetProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager){
		
	}

	void onObjectAdd(Object *obj){};

	void Process(float dt){
		for(cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
			Object *obj = it->second;

			offsetData *data = obj->getProp<offsetData>(Hash::getHash("offsetData"));

			if(data == NULL){
				continue;
			}
			

			
			//return;

			assert(data->parent != NULL);

			vector2 *parentPos = data->parent->getProp<vector2>(Hash::getHash("position"));
			vector2 *pos = obj->getProp<vector2>(Hash::getHash("position"));
			(*pos) = *parentPos + data->posOffset;
			
			util::Angle *facing = obj->getProp<util::Angle>(Hash::getHash("facing"));
			*facing = data->angleOffset;
			/*
			util::Angle *parentFacing = data->parent->getProp<util::Angle>(Hash::getHash("facing"));
			util::Angle *facing = obj->getProp<util::Angle>(Hash::getHash("facing"));
			*facing = *parentFacing + data->angleOffset;
`			*/
		};
	};
};