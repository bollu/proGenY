
#include "PickupProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"

PickupProcessor::PickupProcessor(processMgr &processManager, Settings &settings,
	EventManager &_eventManager) : ObjectProcessor("PickupProcessor"), eventManager(_eventManager) {

};

void pickupCollisionCallback(CollisionData &collision, void *data){

	//EventManager *eventManager = static_cast<EventManager *>(data);
	Object *me = collision.me;
	Object *other = collision.otherObj;

	assert(me != NULL);
	assert(other != NULL);

	PickupData *pickupData = me->getPrimitive<PickupData>(Hash::getHash("PickupData"));
	assert(pickupData != NULL);

	other->sendMessage(pickupData->onPickupEvent, pickupData->eventData, true);
	//HACK
	//send the event with the PickupData
	//eventManager->sendEvent_(pickupData->onPickupEvent, pickupData->eventData);
	
	//after this, the evenData is deleted
	me->Kill();
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
		//delete(data->eventData);
	}
};
