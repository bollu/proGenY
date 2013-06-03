#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"
#include "../../core/Messaging/eventMgr.h"
#include "../../include/Box2D/Box2D.h"

struct moveData{
private:
	bool movingLeft;
	bool movingRight;
	bool jumping;

	bool moveHalted;

	bool onGround;

	friend class groundMoveProcessor;

	vector2 moveImpulse;
	vector2 jumpImpulse;

	float mass;

public:
		
	moveData() : xVel(0), xAccel(0), jumpRange(0), jumpHeight(0), jumpTimeOfFlight(0),
				movingLeft(false), movingRight(false), 
				jumping(false), moveHalted(false), onGround(true){}

	//max vel. with which to move in the x coordinate
	float xVel;
	//amount by which vel can change every second
	float xAccel;
	vector2 movementDamping;

	float jumpRange;
	float jumpHeight;
	float jumpTimeOfFlight;
	
	void setMoveLeft(bool enabled);
	void setMoveRight(bool enabled);

	bool isMovingLeft();
	bool isMovingRight();
	bool isMidJump();
	bool isJumpEnabled();

	//!whether movement should be damped based on stopStrength, and caused to stop
	void setMovementHalt(bool enabled);


	void Jump();
	void resetJump();



};

class groundMoveProcessor : public objectProcessor{
public:
	groundMoveProcessor(b2World &_world) : world(_world){}

	void onObjectAdd(Object *obj);
	void Process(float dt);
private:

	b2World &world;
	vector2 _calcJumpImpulse(moveData *data, float dt);
	
};