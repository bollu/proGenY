#pragma once
#include "phyProcessor.h"
#include <type_traits>


#include "../../World/objContactListener.h"

phyProcessor::phyProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
objectProcessor("phyProcessor") {
	this->view = processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));
	this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"))->getWorld();

	world->SetContactListener(&this->contactListener);
}

void phyProcessor::_onObjectAdd(Object *obj){

	

	phyData *data = obj->getPrimitive<phyData>(Hash::getHash("phyData"));
	vector2* gamePos = obj->getPrimitive<vector2>(Hash::getHash("position"));


	//it doesn't have phyData, so chill and move on
	if(data == NULL){	
		IO::infoLog<<"\n "<<obj->getName()<<" does not have phyData";
		return;

	}
	
	data->bodyDef.position = *gamePos;
	b2Body *body = world->CreateBody(&data->bodyDef);
	data->body = body;
	data->body->SetUserData(obj);
	
	for(auto it = data->fixtureDef.begin(); it != data->fixtureDef.end(); ++it){
		b2FixtureDef fixtureDef = *it;

		b2Fixture *fixture = body->CreateFixture(&fixtureDef);
		data->fixtures.push_back(fixture);
		
		//the fixture def's shape. the fixture def's shape
		//has to be_ created on the heap.
		delete(fixtureDef.shape);
	}

	obj->addProp(Hash::getHash("mass"), new fProp(body->GetMass()) );

}

void phyProcessor::_preProcess(){

	for(Object::cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
		Object *obj = it->second;
		
		assert(obj != NULL);


		phyData *data = obj->getPrimitive<phyData>(Hash::getHash("phyData"));
		if(data == NULL){
			continue;
		}
		
		data->collisions.clear();
	};
	//this->_processContacts();
};

void phyProcessor::_Process(Object *obj, float dt){


	phyData *data = obj->getPrimitive<phyData>(Hash::getHash("phyData"));

	vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));

	util::Angle *angle = obj->getPrimitive<util::Angle>(Hash::getHash("facing"));
	angle->setRad(data->body->GetAngle()); 

	vector2 newPos = vector2::cast(data->body->GetPosition());
	*pos = (newPos);

}


void phyProcessor::_onObjectDeath(Object *obj){
	phyData *data = obj->getPrimitive<phyData>(Hash::getHash("phyData"));


	if(data != NULL){
		IO::infoLog<<"\n\ndestroying body owned by "<<obj->getName();
		world->DestroyBody(data->body);
	}
}

void phyProcessor::_onObjectActivate(Object *obj){
	phyData *data = obj->getPrimitive<phyData>(Hash::getHash("phyData"));
	if(data != NULL){
		
	}
};
void phyProcessor::_onObjectDeactivate(Object *obj){
	phyData *data = obj->getPrimitive<phyData>(Hash::getHash("phyData"));
	if(data != NULL){

	}
};


//-------------------------------------------------------------------

void phyData::addCollision(collisionData &collision){

	this->collisions.push_back(collision);
};

void phyData::removeCollision(Object *obj){
	for(auto it = this->collisions.begin(); it != this->collisions.end(); ++it){
		if(it->otherObj == obj){
			this->collisions.erase(it);
			break;
		}
	};
};


