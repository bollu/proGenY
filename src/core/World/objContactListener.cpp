
#include "objContactListener.h"

#include "../componentSys/processor/PhyProcessor.h"

void objContactListener::_extractObjects(b2Contact *contact, Object **a, Object **b){
	b2Fixture *fixtureA = contact->GetFixtureA();
	b2Fixture *fixtureB = contact->GetFixtureB();

	b2Body *bodyA = fixtureA->GetBody();
	b2Body *bodyB = fixtureB->GetBody();


	*a = static_cast<Object *>(bodyA->GetUserData());
	*b = static_cast<Object *>(bodyB->GetUserData());

	assert(*a != NULL && *b != NULL);
}

void objContactListener::BeginContact(b2Contact* contact){
	this->_handleCollision(true, contact);
};


void objContactListener::EndContact(b2Contact* contact){
	this->_handleCollision(false, contact);
};

void objContactListener::_handleCollision(bool beginHandler, b2Contact *contact){
	Object *a, *b;
	this->_extractObjects(contact, &a, &b);

	PhyData *aPhyData = a->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	PhyData *bPhyData = b->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	assert(aPhyData != NULL && bPhyData != NULL);

	if (aPhyData->collisionHandlers.size() > 0) {
		this->_execHandlers(beginHandler, contact, a, aPhyData, b, bPhyData);
	}

	if (bPhyData->collisionHandlers.size() > 0) {
		this->_execHandlers(beginHandler, contact, b, bPhyData, a, aPhyData);
	}
};


bool objContactListener::_shouldHandleCollision(CollisionHandler *handler, PhyData *otherPhy){
	return handler->otherCollision == otherPhy->collisionType;
};


void objContactListener::_execHandlers(bool beginHandler, b2Contact *contact, Object *me, PhyData *myPhy, Object *other, PhyData *otherPhy){
	CollisionData collisionData;
	collisionData.me = me;
	collisionData.myPhy = myPhy;
	collisionData.otherObj = other;
	collisionData.otherPhy = otherPhy;
	collisionData.contact = contact;

	for(auto collisionHandler : myPhy->collisionHandlers) {
		if(_shouldHandleCollision(&collisionHandler, otherPhy)) {

			if(beginHandler && collisionHandler.onBegin != NULL) {
				collisionHandler.onBegin(collisionData, collisionHandler.data);
			}else if(collisionHandler.onEnd != NULL) {
				collisionHandler.onEnd(collisionData, collisionHandler.data);
			}
		}
	}
};