#pragma once
#include "phyProcessor.h"
#include <type_traits>

void phyProcessor::onObjectAdd(Object *obj){


	phyData *data = obj->getProp<phyData>(Hash::getHash("phyData"));
	vector2* gamePos = obj->getProp<vector2>(Hash::getHash("position"));


	//it doesn't have either, so chill and move on
	if(data == NULL){	
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

void phyProcessor::preProcess(){
	for(cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
		Object *obj = it->second;

		phyData *data = obj->getProp<phyData>(Hash::getHash("phyData"));
		if(data == NULL){
			continue;
		}
		
		data->collisions.clear();
	};
	//this->_processContacts();
};

void phyProcessor::Process(float dt){


	for(cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
		Object *obj = it->second;

		
		phyData *data = obj->getProp<phyData>(Hash::getHash("phyData"));

		if(data == NULL){
			continue;
		}

		vector2 *pos = obj->getProp<vector2>(Hash::getHash("position"));
		util::Angle *angle = obj->getProp<util::Angle>(Hash::getHash("facing"));
		angle->setRad(data->body->GetAngle()); 

		vector2 newPos = vector2::cast(data->body->GetPosition());
		*pos = (newPos);

	};

}


void phyProcessor::postProcess(){

}



void phyProcessor::onObjectRemove(Object *obj){
	phyData *data = obj->getProp<phyData>(Hash::getHash("phyData"));


	if(data != NULL){
		util::msgLog("destroying body");
		world->DestroyBody(data->body);
	}
}

void phyProcessor::_processContacts(){
	for(auto contact = world->GetContactList(); contact != NULL; contact = contact->GetNext()){

		if(!contact->IsTouching()){
			continue;
		}

		b2Fixture *fixtureA = contact->GetFixtureA();
		b2Fixture *fixtureB = contact->GetFixtureB();

		b2Body *bodyA = fixtureA->GetBody();
		b2Body *bodyB = fixtureB->GetBody();


		Object *objA = static_cast<Object *>(bodyA->GetUserData());
		Object *objB = static_cast<Object *>(bodyB->GetUserData());

		assert(objA != NULL && objB != NULL);

		phyData* phyDataA = objA->getProp<phyData>(Hash::getHash("phyData"));
		phyData* phyDataB = objB->getProp<phyData>(Hash::getHash("phyData"));

		assert(phyDataA != NULL && phyDataB != NULL);

		//util::msgLog("collision.\nA:" + objA->getName() + "\nB:" + objB->getName());



	};
}

void phyData::addCollision(collisionData &collision){

	this->collisions.push_back(collision);
};

void phyData::removeCollision(Object *obj){
	for(auto it = this->collisions.begin(); it != this->collisions.end(); ++it){
		if(it->obj == obj){
			this->collisions.erase(it);
			util::msgLog("erased");
			break;
		}
	};
};
