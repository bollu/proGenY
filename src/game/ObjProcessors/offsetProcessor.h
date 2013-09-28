#pragma once
#include "../../core/componentSys/processor/objectProcessor.h"
#include "../../core/math/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/eventMgr.h"

#include "../../core/Rendering/viewProcess.h"
#include "../../core/World/worldProcess.h"
#include "../../core/math/mathUtil.h"


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
