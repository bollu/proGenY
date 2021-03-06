
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
}


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


