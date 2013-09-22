#pragma once
#include "pickupProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"

pickupProcessor::pickupProcessor(processMgr &processManager, Settings &settings,
	eventMgr &_eventManager) : objectProcessor("pickupProcessor"), eventManager(_eventManager) {

};

void pickupProcessor::_Process(Object *obj, float dt){

	pickupData *data = obj->getPrimitive<pickupData>(Hash::getHash("pickupData"));
	phyData *phy = obj->getPrimitive<phyData>(Hash::getHash("phyData"));

	assert(data != NULL && phy != NULL);
	
	for(collisionData &collision : phy->collisions){
		this->_handleCollision(obj, data, collision);
	}

};


void pickupProcessor::_onObjectDeath(Object *obj){
	pickupData *data = obj->getPrimitive<pickupData>(Hash::getHash("pickupData"));

	if(data->eventData != NULL){
		//delete(data->eventData);
	}
};


void pickupProcessor::_handleCollision(Object *obj, pickupData *data, collisionData &collision){
	if(data->hasCollisionType(collision.getCollidedObjectCollision())){
		//send the event with the pickupData
		baseProperty *p = new iProp(10);

		this->eventManager.sendEvent_(data->onPickupEvent, data->eventData);
		//after this, the evenDAta is deleted
		obj->Kill();
	}
};
