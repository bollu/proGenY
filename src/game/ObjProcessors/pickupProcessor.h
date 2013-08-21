#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/Object.h"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"


#include <unordered_set>

struct pickupData {
	const Hash   *onPickupEvent;
	baseProperty *eventData;

	//the collision types of all objects that can pick this up
	std::unordered_set< const Hash * > pickupCollisionTypes;

	void				   addCollisionType ( const Hash *collisionType ){
		this->pickupCollisionTypes.insert( collisionType );
	}


	bool hasCollisionType ( const Hash *collisionType ){
		return (this->pickupCollisionTypes.count( collisionType ) > 0);
	}


	pickupData (){
		this->eventData = NULL;
	}
};


class collisionData;
class pickupProcessor : public objectProcessor
{
public:
	pickupProcessor ( processMgr &processManager, Settings &settings, eventMgr &_eventManager );
	void Process ( float dt );
	void onObjectRemove ( Object *obj );


private:
	eventMgr &eventManager;

	void _handleCollision ( Object *obj, pickupData *data, collisionData &collision );
};