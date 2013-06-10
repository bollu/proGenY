#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"

#include "../../core/Process/viewProcess.h"
#include "../../core/Process/worldProcess.h"



class bulletCollider{
	public:
	bulletCollider(){};
};


struct bulletData{
public:
	vector2 beginVel;
	/*!Angle to face in the beginning in degrees*/
	util::Angle angle;

	std::vector<bulletCollider> colliders;

	const Hash* enemyCollision;

	int damage;

	bulletData() : damage(0){}
};


class bulletProcessor : public objectProcessor{
public:
	bulletProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager){
		this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"))->getWorld();
	}

	void onObjectAdd(Object *obj);
	void Process(float dt);
private:

	b2World *world;

};