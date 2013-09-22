#pragma once
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/objectProcessor.h"
#include "../../core/Process/processMgr.h"
#include "../../core/Process/worldProcess.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"



struct groundMoveData{
private:
	bool movingLeft;
	bool movingRight;
	bool jumping;
	vector2 jumpDir;

	bool onGround;

	friend class groundMoveProcessor;

	vector2 moveImpulse;
	vector2 jumpImpulse;

	float mass;

public:
		
	groundMoveData() : xVel(0), xAccel(0), jumpRange(0), jumpHeight(0), 
				movingLeft(false), movingRight(false), 
				jumping(false), onGround(true){}

	//max vel. with which to move in the x coordinate
	float xVel;
	//amount by which vel can change every second
	float xAccel;
	vector2 movementDamping;


	float jumpRange;
	float jumpHeight;
	
	void setMoveLeft(bool enabled);
	void setMoveRight(bool enabled);

	bool isMovingLeft();
	bool isMovingRight();
	bool isMidJump();
	bool isJumpEnabled();

	void Jump();
	void resetJump();



};

class groundMoveProcessor : public objectProcessor{
public:
	groundMoveProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
		objectProcessor("groundMoveProcessor"){
		this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"))->getWorld();
	}

	void _onObjectAdd(Object *obj);
	void _Process(Object *obj, float dt);
private:

	b2World *world;
	vector2 _calcJumpImpulse(groundMoveData *data, vector2 currentVel, float dt);
	
	bool _shouldProcess(Object *obj){
		return obj->hasProperty("groundMoveData") && obj->requireProperty("phyData");
	};
};
