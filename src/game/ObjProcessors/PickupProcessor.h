#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/componentSys/Object.h"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"


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

struct CollisionData;

class PickupProcessor : public ObjectProcessor{
public:
	PickupProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager);
	
private:
	EventManager &eventManager;
protected:
	void _onObjectAdd(Object *obj);
	void _onObjectDeath(Object *obj);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("PickupData") && obj->requireProperty("PhyData");
	};
};
