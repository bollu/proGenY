#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/math/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"

#include "../../core/Rendering/viewProcess.h"
#include "../../core/World/worldProcess.h"


struct BulletData{
public:
	vector2 beginVel;
	/*!Angle to face in the beginning in degrees*/
	util::Angle angle;

	//!amount by which gravity affects the bullet
	float gravityScale;

	BulletData(){
		gravityScale = 3.0;
	};
	
};


class BulletProcessor : public ObjectProcessor{
public:
	BulletProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
		ObjectProcessor("BulletProcessor"){

			this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));
	}


protected:
	void _onObjectAdd(Object *obj);
	
	bool _shouldProcess(Object *obj){
		return obj->hasProperty("BulletData") && obj->requireProperty("PhyData");
	};

private:
	//void _handleCollision(CollisionData &collision,BulletData *data, Object *obj);
	worldProcess *world;

};
