#pragma once
#include "../../core/math/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/controlFlow/processMgr.h"
#include "../../core/World/worldProcess.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"

struct airMoveData{
private:
	vector2 dir;
	float speed;
	
public:

	void setDir(vector2 dir);
	void setSpeed(float speed);
};


class airMoveProcessor : public ObjectProcessor{
public:
	airMoveProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
			ObjectProcessor("airMoveProcessor"){
		this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));
	}

	
protected:
	void _onObjectAdd(Object *obj);
	void _Process(Object *obj, float dt);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("airMoveData");
	};
	
private:
	worldProcess *world;
};

