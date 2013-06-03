#pragma once
#include "groundMoveProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"

void groundMoveProcessor::Process(float dt){
	for(auto it=  objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;


		moveData *data = obj->getProp<moveData>(Hash::getHash("moveData"));

		if(data == NULL) continue;

		if(data->moveLeft){
			this->_moveLeft(obj, data, dt);
		}
		if(data->moveRight){
			this->_moveRight(obj, data, dt);
		}
		if(data->stopMoving){
			this->_stopMoving(obj, data, dt);
		}
		if(data->jump){
			this->_jump(obj, data, dt);
			data->jump = false;
		}
	};
};
void groundMoveProcessor::_moveLeft(Object *obj, moveData *data, float dt){
	vector2 *impulse = obj->getProp<vector2>(Hash::getHash("impulse"));
	*impulse += vector2(-data->strength.x, 0) * dt; 
};

void groundMoveProcessor::_moveRight(Object *obj, moveData *data, float dt){
	vector2 *impulse = obj->getProp<vector2>(Hash::getHash("impulse"));
	*impulse += vector2(data->strength.x, 0) * dt; 
}; 

void groundMoveProcessor::_jump(Object *obj, moveData *data, float dt){
	vector2 *impulse = obj->getProp<vector2>(Hash::getHash("impulse"));
	*impulse += vector2(0, data->strength.y) * dt ; 
};

void groundMoveProcessor::_stopMoving(Object *obj, moveData *data, float dt){
	const vector2 *vel = obj->getProp<vector2>(Hash::getHash("velocity"));
	vector2 *impulse = obj->getProp<vector2>(Hash::getHash("impulse"));

	*impulse = *vel * dt * (-1); 
	impulse->x *= data->stopStrength.x;
	impulse->y  *= data->stopStrength.y;

};