
#include "PickupProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"

PickupProcessor::PickupProcessor(processMgr &processManager, Settings &settings,
	EventManager &_eventManager) : ObjectProcessor("PickupProcessor"), eventManager(_eventManager) {

};

void pickupCollisionCallback(CollisionData &collision, void *data){
	EventManager *eventManager = static_cast<EventManager *>(data);
	Object *obj = collision.me;
	assert(obj != NULL);

	PickupData *pickupData = obj->getPrimitive<PickupData>(Hash::getHash("PickupData"));
	assert(pickupData != NULL);

	//send the event with the PickupData
	eventManager->sendEvent_(pickupData->onPickupEvent, pickupData->eventData);
	//after this, the evenData is deleted
	obj->Kill();
}

void PickupProcessor::_onObjectAdd(Object *obj){
	PhyData *phy = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	PickupData *data = obj->getPrimitive<PickupData>(Hash::getHash("PickupData"));


	for(const Hash *collisionType : data->pickupCollisionTypes) {
		CollisionHandler pickupCollisionHandler;
		pickupCollisionHandler.otherCollision = collisionType;
		pickupCollisionHandler.onBegin = pickupCollisionCallback;
		pickupCollisionHandler.data = static_cast<void *>(&this->eventManager);

		phy->collisionHandlers.push_back(pickupCollisionHandler);
	}
};

void PickupProcessor::_Process(Object *obj, float dt){

	PickupData *data = obj->getPrimitive<PickupData>(Hash::getHash("PickupData"));
	PhyData *phy = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));

	assert(data != NULL && phy != NULL);
	
	/*
	for(CollisionData &collision : phy->collisions){
		this->_handleCollision(obj, data, collision);
	}*/

};


void PickupProcessor::_onObjectDeath(Object *obj){
	PickupData *data = obj->getPrimitive<PickupData>(Hash::getHash("PickupData"));

	if(data->eventData != NULL){
		//delete(data->eventData);
	}
};

/*
void PickupProcessor::_handleCollision(Object *obj, PickupData *data, CollisionData &collision){
	if(data->hasCollisionType(collision.getCollidedObjectCollision())){
		//send the event with the PickupData
		this->eventManager.sendEvent_(data->onPickupEvent, data->eventData);
		//after this, the evenData is deleted
		obj->Kill();
	}
};
*/