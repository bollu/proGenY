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
	
	bool offsetPos;
	bool offsetAngle;

	Object *parent;

	offsetData() : offsetPos(true), offsetAngle(true){}
};


class offsetProcessor : public objectProcessor{
public:
	offsetProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
	objectProcessor("offsetProcessor"){
	}

protected:
	void _Process(Object *obj, float dt){

		offsetData *data = obj->getPrimitive<offsetData>(Hash::getHash("offsetData"));

		assert(data->parent != NULL);

		if(data->offsetPos){
			vector2 *parentPos = data->parent->getPrimitive<vector2>(Hash::getHash("position"));
			vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));

			(*pos) = *parentPos + data->posOffset;
		}

		if(data->offsetAngle){
			util::Angle *facing = obj->getPrimitive<util::Angle>(Hash::getHash("facing"));
			util::Angle *parentFacing = data->parent->getPrimitive<util::Angle>(
				Hash::getHash("facing"));

			*facing = *parentFacing+ data->angleOffset;
		}
	}
	
	bool _shouldProcess(Object *obj){
		return obj->hasProperty("offsetData");
	};
};
