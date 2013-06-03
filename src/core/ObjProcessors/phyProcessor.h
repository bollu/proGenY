#pragma once
#include "../objectProcessor.h"
#include "../../util/logObject.h"
#include "../../include/Box2D/Box2D.h"
#include "../Process/viewProcess.h"

#include "objContactListener.h"



struct collisionData{
	enum Type{
		onBegin,
		onEnd,
	} type;
	phyData *data;
	Object *obj;	
};


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
	b2World &world;
	viewProcess &view;
	objContactListener contactListener;

	void _processContacts();
public:
	phyProcessor(b2World &_world, viewProcess &_view) : world(_world), view(_view){
		world.SetContactListener(&this->contactListener);
	}

	void onObjectAdd(Object *obj);

	void preProcess();
	void Process(float dt);
	void postProcess();
	void onObjectRemove(Object *obj);
};

