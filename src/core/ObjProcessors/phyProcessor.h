#pragma once
#include "../objectProcessor.h"
#include "../../util/logObject.h"
#include "../../include/Box2D/Box2D.h"
#include "../Process/viewProcess.h"

struct phyData{
	b2BodyDef bodyDef;
	std::vector<b2FixtureDef> fixtureDef;

	//don't modify this
	b2Body *body;
	std::vector<b2Fixture*>fixtures;

	vector2 maxVel;
	bool velClamped;
};


class phyProcessor : public objectProcessor{
private:
	b2World &world;
	viewProcess &view;

public:
	phyProcessor(b2World &_world, viewProcess &_view) : world(_world), view(_view){}

	void onObjectAdd(Object *obj);

	void Process(float dt);

	void onObjectRemove(Object *obj);
};

