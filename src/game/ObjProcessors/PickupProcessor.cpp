
#include "PickupProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"

PickupProcessor::PickupProcessor(processMgr &processManager, Settings &settings,
	EventManager &_eventManager) : ObjectProcessor("PickupProcessor"), eventManager(_eventManager) {

};

bool isValidCollision(std::unordered_set<const Hash*> &pickupCollisions, const Hash *objCollisionType) {
	return pickupCollisions.find(objCollisionType) != pickupCollisions.end();
}

void pickupCollisionCallback(CollisionData &collision, void *data){

	//EventManager *eventManager = static_cast<EventManager *>(data);
	Object *me = collision.me;
	Object *other = collision.otherObj;

	assert(me != NULL);
	assert(other != NULL);

	PickupData *pickupData = me->getPrimitive<PickupData>(Hash::getHash("PickupData"));
	assert(pickupData != NULL);

	if (isValidCollision(pickupData->pickupCollisionTypes, collision.otherPhy->collisionType)) {
		other->sendMessage(pickupData->onPickupEvent, pickupData->eventData, true);
		me->Kill();
	};
}

void PickupProcessor::_onObjectAdd(Object *obj){
	PhyData *phy = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	PickupData *data = obj->getPrimitive<PickupData>(Hash::getHash("PickupData"));


		CollisionHandler pickupCollisionHandler;
		pickupCollisionHandler.otherCollision = CollisionHandler::ALL_COLLISIONS;
		
		pickupCollisionHandler.onBegin = pickupCollisionCallback;
		phy->collisionHandlers.push_back(pickupCollisionHandler);
};


void PickupProcessor::_onObjectDeath(Object *obj){
	PickupData *data = obj->getPrimitive<PickupData>(Hash::getHash("PickupData"));

	if(data->eventData != NULL){
		delete(data->eventData);
	}
};
