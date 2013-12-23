
#include "PhyProcessor.h"
#include <type_traits>


#include "../../World/objContactListener.h"

//this is sort of a HACK. need to fix this
worldProcess *PhyProcessor::world;

PhyProcessor::PhyProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
ObjectProcessor("PhyProcessor") {
	this->view = processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));
	this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));

	world->setContactListener(&this->contactListener);
}



PhyData PhyProcessor::createPhyData(b2BodyDef *bodyDef, b2FixtureDef fixtures[], unsigned int numFixtures){
	PhyData phyData; 

	//phy.bodyDef.type = b2_dynamicBody;

	IO::infoLog<<"\nnumFixtures: "<<numFixtures<<IO::flush;

	b2Body *body = world->createBody(bodyDef);
	phyData.body = body;

	for(int i = 0; i < numFixtures; i++){

		b2FixtureDef *fixtureDef = &fixtures[i];
		b2Fixture *fixture = body->CreateFixture(fixtureDef);
		//phyData->fixtures.push_back(fixture);
		
		//the fixture def's shape. the fixture def's shape
		//has to be_ created on the heap.
		//delete(fixtureDef.shape);
	}

	return phyData;
};

void PhyProcessor::_onObjectAdd(Object *obj){
	
	PhyData *phyData = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	vector2* gamePos = obj->getPrimitive<vector2>(Hash::getHash("position"));

	b2Body *body = phyData->body;
	body->SetTransform(gamePos->cast<b2Vec2>(), 0);
	body->SetUserData(obj);
	obj->addProp(Hash::getHash("mass"), new fProp(body->GetMass()) );

	//it doesn't have PhyData, so chill and move on
	/*if(data == NULL){	
		IO::infoLog<<"\n "<<obj->getName()<<" does not have PhyData";
		return;

	}*/

	
	/*data->bodyDef.position = *gamePos;
	b2Body *body = world->createBody(&data->bodyDef);
	data->body = body;
	data->body->SetUserData(obj);
	
	for(auto it = data->fixtureDef.begin(); it != data->fixtureDef.end(); ++it){
		b2FixtureDef fixtureDef = *it;

		b2Fixture *fixture = body->CreateFixture(&fixtureDef);
		data->fixtures.push_back(fixture);
		
		//the fixture def's shape. the fixture def's shape
		//has to be_ created on the heap.
		delete(fixtureDef.shape);
	}*/



}

void PhyProcessor::_preProcess(){

	for(Object::cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
		Object *obj = it->second;
		
		assert(obj != NULL);


		PhyData *data = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
		if(data == NULL){
			continue;
		}
		
		//data->collisions.clear();
	};
	//this->_processContacts();
};

void PhyProcessor::_Process(Object *obj, float dt){


	PhyData *data = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));

	util::Angle *angle = obj->getPrimitive<util::Angle>(Hash::getHash("facing"));
	angle->setRad(data->body->GetAngle()); 

	*pos= vector2::cast(data->body->GetPosition());


}


void PhyProcessor::_onObjectDeath(Object *obj){
	PhyData *data = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));


	if(data != NULL){
		IO::infoLog<<"\n\ndestroying body owned by "<<obj->getName();
		world->destroyBody(data->body);
	}
}

void PhyProcessor::_onObjectActivate(Object *obj){
	PhyData *data = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	if(data != NULL){
		
	}
};
void PhyProcessor::_onObjectDeactivate(Object *obj){
	PhyData *data = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	if(data != NULL){

	}
};


//-------------------------------------------------------------------
/*
void PhyData::addCollision(CollisionData &collision){

	this->collisions.push_back(collision);
};

void PhyData::removeCollision(Object *obj){
	for(auto it = this->collisions.begin(); it != this->collisions.end(); ++it){
		if(it->otherObj == obj){
			this->collisions.erase(it);
			break;
		}
	};
};*/


