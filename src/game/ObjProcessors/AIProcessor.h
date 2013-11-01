#pragma once
#include "../../core/componentSys/processor/objectProcessor.h"
#include "../../core/math/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/AI/baseAI.h"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/eventMgr.h"

#include "../../core/Rendering/viewProcess.h"
#include "../../core/World/worldProcess.h"



struct AIData{
	
};


class AIProcessor : public objectProcessor{
public:
	AIProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
		objectProcessor("AIProcessor"){
			this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));
	}

protected:
	void onObjectAdd(Object *obj){};
	void _Process(Object *obj, float dt){};

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("AIData");
	};

private:
	worldProcess *world;
};





