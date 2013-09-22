#pragma once
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/objectProcessor.h"
#include "../../core/Process/processMgr.h"
#include "../../core/Process/worldProcess.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"

struct airMoveData{
private:
	vector2 dir;
	float speed;
	
public:

	void setDir(vector2 dir);
	void setSpeed(float speed);
};


class airMoveProcessor : public objectProcessor{
public:
	airMoveProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
			objectProcessor("airMoveProcessor"){
		this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"))->getWorld();
	}

	
protected:
	void _onObjectAdd(Object *obj);
	void _Process(Object *obj, float dt);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("airMoveData");
	};
	
private:
	b2World *world;
};

