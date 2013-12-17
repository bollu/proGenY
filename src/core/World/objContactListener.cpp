#pragma once
#include "objContactListener.h"

#include "../componentSys/processor/PhyProcessor.h"

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

};
void objContactListener::EndContact(b2Contact* contact){

	this->_handleCollision(collisionData::Type::onEnd, contact);
	
};




collisionData objContactListener::_fillCollisionData(b2Contact *contact,
  Object *me, Object *other, PhyData *myPhy, PhyData *otherPhy){

	collisionData collision;

	collision.myPhy = myPhy;
	collision.otherPhy = otherPhy;
	collision.otherObj = other;

	collision.myApproachVel = vector2::cast(myPhy->body->GetLinearVelocity());

	b2Manifold *localManifold = contact->GetManifold();
	b2WorldManifold worldManifold;

	 contact->GetWorldManifold(&worldManifold);
	//if(collision.normal == zeroVector){

	/*1) if the point exists, use it.
	the point is the most accurate way of figuring out the collision
	  
	2)if no point, use the collision normal  
	  this is *NOT* the actual collision normal. it is the vector that points
	  in the direction such that the two bodies can be seperated with the most 
	  ease in this direction.

	  3) if neither, make a sucky ballpark estimation based to relative velocities 

	  */
	if(vector2::cast(worldManifold.normal) != zeroVector){
		collision.normal = vector2::cast(worldManifold.normal);
	}
	else if(vector2::cast(localManifold->localNormal) != zeroVector){
	
		collision.normal = vector2::cast(localManifold->localNormal);
	}
	else if(vector2::cast(localManifold->localPoint) != zeroVector){
		collision.normal = vector2::cast(localManifold->localPoint);
	}
	
	 else if(localManifold->pointCount > 0){
		collision.normal = vector2::cast(localManifold->localPoint);
	} 
	else{
		collision.normal = nullVector;
	}
	

	collision.normal.Normalize();
	 
	return collision;
	
};	



void objContactListener::_handleCollision(collisionData::Type type, b2Contact *contact){
	
	collisionData collision;
	Object *a, *b;
	
	this->_extractPhyData(contact, &a, &b);
	
	assert(a != NULL && b != NULL);

	PhyData *aPhyData = a->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	PhyData *bPhyData = b->getPrimitive<PhyData>(Hash::getHash("PhyData"));
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


const Hash *collisionData::getCollidedObjectCollision(){
	return this->otherPhy->collisionType;
};

