#pragma once
#include "../objectProcessor.h"
#include "../../util/logObject.h"
#include "../../include/Box2D/Box2D.h"

class phyProcessor : public objectProcessor{
private:
	b2World &world;
public:
	phyProcessor(b2World &_world) : world(_world){}

	void onObjectAdd(Object *obj){
		auto b2BodyDefProp = obj->getProp<b2BodyDef>(Hash::getHash("b2BodyDef"));
		auto b2FixtureDefProp = obj->getProp<b2FixtureDef>(Hash::getHash("b2FixtureDef"));
		auto posProp = obj->getProp<vector2>(Hash::getHash("position"));


		//it dosen't have either, so chill and move on
		if(b2BodyDefProp == NULL && b2FixtureDefProp == NULL){
			return;
		}

		//it has one, but not the other, time to screw it
		if(b2BodyDefProp == NULL || b2FixtureDefProp == NULL){
			util::msgLog(obj->getName() + " does not posess either b2BodyDefProp or b2FixtureDefProp", 
				util::logLevel::logLevelError);
		}


		auto bodyDef = b2BodyDefProp->getVal();
		auto fixtureDef = b2FixtureDefProp->getVal();

		bodyDef.position = posProp->getVal();

		b2Body *body = world.CreateBody(&bodyDef);
		b2Fixture *fixture = body->CreateFixture(&fixtureDef);


		obj->addProp(Hash::getHash("b2Body"), new Prop<b2Body *>(body));
		obj->addProp(Hash::getHash("b2Fixture"), new Prop<b2Fixture *>(fixture));
		
	}

	void Process(){

		
		for(cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
			Object *obj = it->second;

			//you're guarenteed to have the position property
			auto posProp = obj->getProp<vector2>(Hash::getHash("position"));

			auto b2BodyProp = obj->getProp<b2Body *>(Hash::getHash("b2Body"));

			if(b2BodyProp == NULL){
				return;
			}
		
			vector2 newPos = vector2::cast( b2BodyProp->getVal()->GetPosition() );
			posProp->setVal(newPos);
		}
	}
};

