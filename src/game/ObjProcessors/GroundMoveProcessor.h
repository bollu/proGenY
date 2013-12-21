#pragma once
#include "../../core/math/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/controlFlow/processMgr.h"
#include "../../core/World/worldProcess.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"



struct groundMoveData{
private:
	friend class groundMoveProcessor;

	bool movingLeft;
	bool movingRight;
	bool jumping;
	
	bool onGround;



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
	
	/*
	void setMoveLeft(bool enabled);
	void setMoveRight(bool enabled);

	bool isMovingLeft();
	bool isMovingRight();
	bool isMidJump();
	bool isJumpEnabled();

	void Jump();
	void resetJump();*/

	//collision type of surface that should allow jumping from and landing on
	const Hash *jumpSurfaceCollision = NULL;
};

class groundMoveProcessor : public ObjectProcessor{
public:
	groundMoveProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
		ObjectProcessor("groundMoveProcessor"){
			this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));
	}

	void _onObjectAdd(Object *obj);
	void _Process(Object *obj, float dt);

private:
	worldProcess *world;
	vector2 _calcJumpImpulse(groundMoveData *data, vector2 currentVel, float dt);
	
	bool _shouldProcess(Object *obj){
		return obj->hasProperty("groundMoveData") && obj->requireProperty("PhyData");
	};

	void _ProcessEvents(Object *obj, groundMoveData *data);
};
