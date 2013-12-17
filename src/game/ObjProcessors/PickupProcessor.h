#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/componentSys/Object.h"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/eventMgr.h"


#include <unordered_set>

struct PickupData{
	const Hash *onPickupEvent;
	baseProperty *eventData;

	//the collision types of all objects that can pick this up
	std::unordered_set<const Hash *> pickupCollisionTypes;
	
	void addCollisionType(const Hash *collisionType){
		this->pickupCollisionTypes.insert(collisionType);
	}

	bool hasCollisionType(const Hash *collisionType){
		return this->pickupCollisionTypes.count(collisionType) > 0;
	}

	PickupData(){
		this->eventData = NULL;
	}


};

class collisionData;

class PickupProcessor : public ObjectProcessor{
public:
	PickupProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager);
	
private:
	eventMgr &eventManager;
	void _handleCollision(Object *obj, PickupData *data, collisionData &collision);


protected:
	void _Process(Object *obj, float dt);
	void _onObjectDeath(Object *obj);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("PickupData") && obj->requireProperty("PhyData");
	};
};
