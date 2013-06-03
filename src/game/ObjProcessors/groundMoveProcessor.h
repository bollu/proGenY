#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"
#include "../../core/Messaging/eventMgr.h"


struct moveData{
	bool moveLeft;
	bool moveRight;
	
	bool stopMoving;

	//the amount by which the stopping impulse should be scaled in the x and y axes
	vector2 stopStrength;

	//the amount by which the movement impulse should be scaled in the x and y axes
	vector2 strength;

	bool jump;

	moveData() : moveLeft(false), moveRight(false), jump(false), stopMoving(false){};

};

class groundMoveProcessor : public objectProcessor{
public:
	void Process(float dt);
private:
	void _moveLeft(Object *o, moveData *data, float dt);
	void _moveRight(Object *o, moveData *data, float dt);
	void _jump(Object *o, moveData *data, float dt);
	void _stopMoving(Object *o, moveData *data, float dt);

};