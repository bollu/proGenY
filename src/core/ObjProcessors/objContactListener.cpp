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
	
	this->_handleCollision(collisionData::Type::onBegin, contact);
	/*
	Object *a, *b;
	b2WorldManifold worldManifold;

	collisionData collision;


	this->_extractPhyData(contact, &a, &b);
	contact->GetWorldManifold(&worldManifold);


	assert(a != NULL && b != NULL);

	phyData *aPhyData = a->getProp<phyData>(Hash::getHash("phyData"));
	phyData *bPhyData = b->getProp<phyData>(Hash::getHash("phyData"));


	assert(aPhyData != NULL && bPhyData != NULL);

	collision.myPhy = aPhyData;
	collision.otherPhy = bPhyData;
	collision.otherObj = b;
	collision.type = collisionData::Type::onBegin;
	collision.normal = vector2::cast(worldManifold.normal);
	aPhyData->addCollision(collision);

	collision.myPhy = bPhyData;
	collision.otherPhy = aPhyData;
	collision.otherObj = a;
	collision.type = collisionData::Type::onBegin;
	collision.normal = vector2::cast(worldManifold.normal);

	bPhyData->addCollision(collision);
	*/

};
void objContactListener::EndContact(b2Contact* contact){

	this->_handleCollision(collisionData::Type::onEnd, contact);
	/*
	Object *a, *b;
	collisionData collision;
	b2WorldManifold worldManifold;

	this->_extractPhyData(contact, &a, &b);
	contact->GetWorldManifold(&worldManifold);

	assert(a != NULL && b != NULL);
	
	phyData *aPhyData = a->getProp<phyData>(Hash::getHash("phyData"));
	phyData *bPhyData = b->getProp<phyData>(Hash::getHash("phyData"));

	assert(aPhyData != NULL && bPhyData != NULL);

	collision.myPhy = aPhyData;
	collision.otherPhy = bPhyData;
	collision.otherObj = b;
	collision.type = collisionData::Type::onEnd;
	collision.normal = vector2::cast(worldManifold.normal);
	

	aPhyData->addCollision(collision);

	collision.myPhy = bPhyData;
	collision.otherPhy = aPhyData;
	collision.otherObj = a;
	collision.type = collisionData::Type::onEnd;
	collision.normal = vector2::cast(worldManifold.normal);
	collision.type = collisionData::Type::onEnd;

	//no need to reset type
	//no need to recompute normal;

	bPhyData->addCollision(collision);*/
	
};




collisionData objContactListener::_fillCollisionData(b2Contact *contact,
  Object *me, Object *other, phyData *myPhy, phyData *otherPhy){

	//assert(contact->IsTouching());

	collisionData collision;

	collision.myPhy = myPhy;
	collision.otherPhy = otherPhy;
	collision.otherObj = other;


	b2Manifold *localManifold = contact->GetManifold();

	//if(collision.normal == zeroVector){

	/*1) if the point exists, use it.
	the point is the most accurate way of figuring out the collision
	  
	2)if no point, use the collision normal  
	  this is *NOT* the actual collision normal. it is the vector that points
	  in the direction such that the two bodies can be seperated with the most 
	  ease in this direction.

	  3) if neither, make a sucky ballpark estimation based to relative velocities 

	  */
	
	if(vector2::cast(localManifold->localNormal) != zeroVector){
		util::infoLog("normal exists");
		collision.normal = vector2::cast(localManifold->localNormal);
	}
	else if(vector2::cast(localManifold->localPoint) != zeroVector){
		util::infoLog("point exists");
		collision.normal = vector2::cast(localManifold->localPoint);
	}
	 else if(localManifold->pointCount > 0){
		util::infoLog("POINT ASIDJUQWR exist");

		collision.normal = vector2::cast(localManifold->localPoint);
	}
	// this appears to be EXTREMELY inaccurate
	

	else{
		util::warningLog("creating a hypothetical normal. use with caution");

		b2WorldManifold worldManifold;
		contact->GetWorldManifold(&worldManifold);

		vector2 myVel = vector2::cast(myPhy->body->
			GetLinearVelocityFromWorldPoint( worldManifold.points[0] ));

		vector2 otherVel= vector2::cast(otherPhy->body->
			GetLinearVelocityFromWorldPoint( worldManifold.points[0] ));

		collision.normal = (myVel - otherVel).Normalize();
		
	}
	

	return collision;
	
};	



void objContactListener::_handleCollision(collisionData::Type type, b2Contact *contact){
	collisionData collision;
	Object *a, *b;
	
	this->_extractPhyData(contact, &a, &b);
	
	assert(a != NULL && b != NULL);

	phyData *aPhyData = a->getProp<phyData>(Hash::getHash("phyData"));
	phyData *bPhyData = b->getProp<phyData>(Hash::getHash("phyData"));
	assert(aPhyData != NULL && bPhyData != NULL);

	/* a to b */
	collision = this->_fillCollisionData(contact, a, b, aPhyData, bPhyData);
	collision.type = type;

	aPhyData->addCollision(collision);

	/*b to a*/
	collision = this->_fillCollisionData(contact, b, a, bPhyData, aPhyData);
	collision.type = type;
	bPhyData->addCollision(collision);
};	
