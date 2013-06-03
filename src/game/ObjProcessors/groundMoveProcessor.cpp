#pragma once
#include "groundMoveProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include <algorithm>


void groundMoveProcessor::onObjectAdd(Object *obj){
	moveData *data = obj->getProp<moveData>(Hash::getHash("moveData"));
	float *mass = obj->getProp<float>(Hash::getHash("mass"));

	if(data == NULL) return;
	assert(mass != NULL);

	data->mass = *mass;
	data->moveImpulse.x = *mass * abs(data->xVel);

	float g = this->world.GetGravity().y;


	//u cos theta= R / (t)
	data->jumpImpulse.x = *mass * data->jumpRange / (data->jumpTimeOfFlight);
	//u sin theta = sqrt(2 * g * h)
	data->jumpImpulse.y = *mass *  sqrt(abs(2 * g * data->jumpHeight));


};

void groundMoveProcessor::Process(float dt){
	for(auto it=  objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;


		moveData *data = obj->getProp<moveData>(Hash::getHash("moveData"));

		if(data == NULL) continue;


		phyData *physicsData = obj->getProp<phyData>(Hash::getHash("phyData"));


		b2Body *body = physicsData->body;


		
		assert(physicsData != NULL);
		assert(body != NULL);

		
		vector2 impulse = vector2(0, 0);
		vector2 vel = vector2::cast(body->GetLinearVelocity());

		float desiredVelX = 0.0f;
		float currentVelX = vel.x;
		

		if(data->jumping && data->onGround){
			//set velocity to 0 for a perfect parabola
			body->SetLinearVelocity(vector2(0, 0));

			impulse += this->_calcJumpImpulse(data, dt);
			data->onGround = false;
			
		}

		else if(data->movingLeft){
			desiredVelX = std::max(-data->xVel, 
				currentVelX - data->xAccel);
		}

		else if(data->movingRight){
			desiredVelX = std::min(data->xVel, 
				currentVelX + data->xAccel);
		}


		//you're still moving, AND you're not jumping
		if(!data->moveHalted && !data->jumping){

	    	//delta v = desired - current	
			float dvX = desiredVelX - currentVelX;
	    	//impulse = m * a * t = m * delta v
	    	float impulseX = data->mass * dvX; //disregard time factor

	    	impulse.x += impulseX;
	    }

	    //you're not jumping in the air
	    if(!data->jumping){
	    	float frictionX = data->mass * currentVelX * data->movementDamping.x;
	    	impulse.x -= frictionX;
	    };

	    body->ApplyLinearImpulse(impulse, body->GetWorldCenter());

	};
};

vector2 groundMoveProcessor::_calcJumpImpulse(moveData *data, float dt){

	util::msgLog("jumping");
	
	vector2 impulse;

	if(data->moveHalted){
		impulse.y = data->jumpImpulse.y;
	}
	else{

		if(data->movingRight){
			impulse.x = data->jumpImpulse.x;
		}else{
			impulse.x = -data->jumpImpulse.x;
		}
		impulse.y += data->jumpImpulse.y;
	}

	return impulse;
};


void moveData::setMoveLeft(bool enabled){
	this->movingLeft = enabled;
};

void moveData::setMoveRight(bool enabled){
	this->movingRight = enabled;
};

void moveData::setMovementHalt(bool enabled){
	this->moveHalted = enabled;
};

void moveData::Jump(){

	if(this->onGround){
		this->jumping = true;
	};
};

void moveData::resetJump(){

	this->onGround = true;
	this->jumping = false;
};

bool moveData::isMovingLeft(){
	return this->movingLeft;
};

bool moveData::isMovingRight(){
	return this->movingRight;
};

bool moveData::isMidJump(){
	//if you're able to jump, you're on the ground :D
	return this->jumping;
};
bool moveData::isJumpEnabled(){
	return !this->jumping;
};
