#pragma once
#include "pickupProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"

pickupProcessor::pickupProcessor(processMgr &processManager, Settings &settings,
 eventMgr &_eventManager) : eventManager(_eventManager) {

};

void pickupProcessor::Process(float dt){

	

	for(auto it=  objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;

		PickupData *data = obj->getProp<PickupData>(Hash::getHash("PickupData"));
		
		if(data == NULL){
			continue;
		}	

		phyData *phy = obj->getProp<phyData>(Hash::getHash("phyData"));
		assert(phy != NULL);

		for(collisionData &collision : phy->collisions){
			this->_handleCollision(obj, data, collision);

		}

	}
};


void pickupProcessor::onObjectRemove(Object *obj){
	PickupData *data = obj->getProp<PickupData>(Hash::getHash("PickupData"));
	
	if(data == NULL){
		return;
	}

	if(data->eventData != NULL){
		//delete(data->eventData);
	}
};


void pickupProcessor::_handleCollision(Object *obj, PickupData *data, collisionData &collision){
	if(data->hasCollisionType(collision.getCollidedObjectCollision())){
		//send the event with the PickupData
		baseProperty *p = new iProp(10);

		this->eventManager.sendEvent_(data->onPickupEvent, data->eventData);
		//after this, the evenDAta is deleted
		obj->Kill();
	}
};