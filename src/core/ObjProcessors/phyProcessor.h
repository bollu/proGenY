#pragma once
#include "../objectProcessor.h"
#include "../../util/logObject.h"

#include "../Process/viewProcess.h"
#include "../Process/worldProcess.h"

#include "../Process/processMgr.h"
#include "../Settings.h"
#include "../Messaging/eventMgr.h"

#include "objContactListener.h"

struct collisionData{
	enum Type{
		onBegin,
		onEnd,
	} type;
	
	phyData *data;
	Object *obj;	
};




//ALYWAS CREATE THE FIXTURE DEF's SHAPE ON THE STACK
struct phyData{
	b2BodyDef bodyDef;
	std::vector<b2FixtureDef> fixtureDef;

	//don't modify this
	b2Body *body;
	std::vector<b2Fixture*>fixtures;

	vector2 maxVel;
	bool velClamped;

	const Hash* collisionType;

	//a map between collision type and collisions
	 std::vector<collisionData> collisions;

	 void addCollision(collisionData &collision);
	 void removeCollision(Object *obj);
};




class phyProcessor : public objectProcessor{
private:
	b2World *world;
	viewProcess *view;
	objContactListener contactListener;

	void _processContacts();
public:
	phyProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager) 
	{
		this->view = processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));
		this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"))->getWorld();

		world->SetContactListener(&this->contactListener);
	}

	void onObjectAdd(Object *obj);

	void preProcess();
	void Process(float dt);
	void postProcess();
	void onObjectRemove(Object *obj);
};

