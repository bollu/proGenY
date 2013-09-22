#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/AI/baseAI.h"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"

#include "../../core/Process/viewProcess.h"
#include "../../core/Process/worldProcess.h"



struct AIData{
	
};


class AIProcessor : public objectProcessor{
public:
	AIProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
		objectProcessor("AIProcessor"){
		this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"))->getWorld();
	}

protected:
	void onObjectAdd(Object *obj){};
	void _Process(Object *obj, float dt){};

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("AIData");
	};

private:
	b2World *world;
};





