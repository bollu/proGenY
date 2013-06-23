#pragma once
#include "objContactListener.h"
#include "phyProcessor.h"


void objContactListener::_extractPhyData(b2Contact *contact, Object **a, Object **b){
	b2Fixture *fixtureA = contact->GetFixtureA();
	b2Fixture *fixtureB = contact->GetFixtureB();

	b2Body *bodyA = fixtureA->GetBody();
	b2Body *bodyB = fixtureB->GetBody();


	*a = static_cast<Object *>(bodyA->GetUserData());
	*b = static_cast<Object *>(bodyB->GetUserData());

	assert(*a != NULL && *b != NULL);
}

void objContactListener::BeginContact(b2Contact* contact){
	
	Object *a, *b;
	collisionData collision;

	this->_extractPhyData(contact, &a, &b);


	assert(a != NULL && b != NULL);

	phyData *aPhyData = a->getProp<phyData>(Hash::getHash("phyData"));
	phyData *bPhyData = b->getProp<phyData>(Hash::getHash("phyData"));


	assert(aPhyData != NULL && bPhyData != NULL);

	collision.data = bPhyData;
	collision.obj = b;
	collision.type = collisionData::Type::onBegin;

	aPhyData->addCollision(collision);

	collision.data = aPhyData;
	collision.obj = a;
	collision.type = collisionData::Type::onBegin;

	bPhyData->addCollision(collision);

};
void objContactListener::EndContact(b2Contact* contact){
	
	Object *a, *b;
	collisionData collision;

	this->_extractPhyData(contact, &a, &b);

	assert(a != NULL && b != NULL);
	
	phyData *aPhyData = a->getProp<phyData>(Hash::getHash("phyData"));
	phyData *bPhyData = b->getProp<phyData>(Hash::getHash("phyData"));

	assert(aPhyData != NULL && bPhyData != NULL);

	collision.data = bPhyData;
	collision.obj = b;
	collision.type = collisionData::Type::onEnd;
	
	

	aPhyData->addCollision(collision);

	collision.data = aPhyData;
	collision.obj = a;
	collision.type = collisionData::Type::onEnd;

	bPhyData->addCollision(collision);
	
};