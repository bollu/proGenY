#pragma once
#include "PickupProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"

PickupProcessor::PickupProcessor(processMgr &processManager, Settings &settings,
	eventMgr &_eventManager) : ObjectProcessor("PickupProcessor"), eventManager(_eventManager) {

};

void PickupProcessor::_Process(Object *obj, float dt){

	PickupData *data = obj->getPrimitive<PickupData>(Hash::getHash("PickupData"));
	PhyData *phy = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));

	assert(data != NULL && phy != NULL);
	
	for(collisionData &collision : phy->collisions){
		this->_handleCollision(obj, data, collision);
	}

};


void PickupProcessor::_onObjectDeath(Object *obj){
	PickupData *data = obj->getPrimitive<PickupData>(Hash::getHash("PickupData"));

	if(data->eventData != NULL){
		//delete(data->eventData);
	}
};


void PickupProcessor::_handleCollision(Object *obj, PickupData *data, collisionData &collision){
	if(data->hasCollisionType(collision.getCollidedObjectCollision())){
		//send the event with the PickupData
		baseProperty *p = new iProp(10);

		this->eventManager.sendEvent_(data->onPickupEvent, data->eventData);
		//after this, the evenDAta is deleted
		obj->Kill();
	}
};
