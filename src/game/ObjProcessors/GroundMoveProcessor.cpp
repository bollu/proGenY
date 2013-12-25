
#include "GroundMoveProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"
#include <algorithm>


void resetJump(CollisionData &collision, void *data){
	bool *onGround = static_cast<bool *>(data);
	*onGround = true;
}

vector2 generateJumpImpulse(float g, float mass, float range, float height) {
	vector2 jumpImpulse;

	jumpImpulse.y = mass *  sqrt(abs(2 * g * height));
	float halfTime = std::abs(jumpImpulse.y / g);
	//u cos theta= R / (t)
	jumpImpulse.x = mass * std::abs(range / (halfTime * 2));

	return jumpImpulse;
}

void groundMoveProcessor::_onObjectAdd(Object *obj){
	groundMoveData *data = obj->getPrimitive<groundMoveData>(Hash::getHash("groundMoveData"));

	float *mass = obj->getPrimitive<float>(Hash::getHash("mass"));
	assert(mass != NULL);

	data->mass = *mass;
	data->moveImpulse.x = *mass * abs(data->xVel);

	data->jumpImpulse = generateJumpImpulse(this->world->getGravity().y, data->mass, data->jumpRange, data->jumpHeight);


	assert(data->jumpSurfaceCollision != NULL);
	PhyData *phyData =  obj->getPrimitive<PhyData>("PhyData");

	CollisionHandler resetJumpHandler;
	resetJumpHandler.otherCollision = data->jumpSurfaceCollision;
	resetJumpHandler.onBegin = resetJump;
	resetJumpHandler.data = static_cast<void *>(&data->onGround);

	phyData->collisionHandlers.push_back(resetJumpHandler);

};

void groundMoveProcessor::_ProcessEvents(Object *obj, groundMoveData *data) {
	bool *moveLeft = obj->getMessage<bool>(Hash::getHash("moveLeft"));
	bool *moveRight = obj->getMessage<bool>(Hash::getHash("moveRight"));
	void *Jump = obj->getMessage<void>(Hash::getHash("Jump"));
	
	if(moveLeft != NULL){
		data->movingLeft = *moveLeft;
	}

	if(moveRight != NULL){
		data->movingRight = *moveRight;
	}

	//HACK
	if(Jump != NULL /*&& data->onGround*/){
		data->jumping = true;
	}
}

void groundMoveProcessor::_Process(Object *obj, float dt){
	groundMoveData *data = obj->getPrimitive<groundMoveData>(Hash::getHash("groundMoveData"));
	PhyData *physicsData = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));

	b2Body *body = physicsData->body;


	_ProcessEvents(obj, data);

	vector2 impulse = vector2(0, 0);
	vector2 vel = vector2::cast(body->GetLinearVelocity());

	float desiredVelX = 0.0f;
	float currentVelX = vel.x;

	
	//you're gonna start jumping
	if(data->jumping){

		//set velocity to 0 for a perfect parabola
		body->SetLinearVelocity(vector2(0, 0));

		impulse += this->_calcJumpImpulse(data, vel, dt);
		
		data->onGround = false;
		data->jumping = false;

	}

	if(data->movingLeft && !data->jumping){
		desiredVelX = std::max(-data->xVel, 
			currentVelX - data->xAccel);
	}

	if(data->movingRight && !data->jumping){
		desiredVelX = std::min(data->xVel, 
			currentVelX + data->xAccel);
	}



	//you're still moving
	//apply the required impulse
	if((data->movingLeft || data->movingRight) && !data->jumping ){
		 //delta v = desired - current	
		float dvX = desiredVelX - currentVelX;
		//impulse = m * a * t = m * delta v
		float impulseX = data->mass * dvX; //disregard time factor
		impulse.x += impulseX;

	};

	//you're not jumping in the air, but you're not actively moving either apply friction
	if(!data->jumping){
		float frictionX = data->mass * currentVelX * data->movementDamping.x;
		impulse.x -= frictionX;

		//body->SetAngularVelocity(-currentVelX  * 0.5);
	};

	body->ApplyLinearImpulse(impulse, body->GetWorldCenter(), true);
};

vector2 groundMoveProcessor::_calcJumpImpulse(groundMoveData *data, vector2 currentVel, float dt){

	vector2 impulse;

	const float angledJumpThreshold = 0.1;


	//if(!(data->movingLeft || data->movingRight)){
	//if(abs(currentVel.x)  < angledJumpThreshold){
	if(true){

		float absCurrentVelX = abs(currentVel.x);

		impulse.y = data->jumpImpulse.y;


		vector2 additionalImpulse;
		additionalImpulse.x = std::min(absCurrentVelX, data->jumpImpulse.x);
		additionalImpulse.x *= currentVel.x < 0 ? -1 : 1;

		impulse += additionalImpulse;
	}
	/*
	else{

		float jumpFraction = std::abs(currentVel.x / data->jumpImpulse.x);
		jumpFraction = jumpFraction < 1 ? jumpFraction : 1;  


		//HACK
		jumpFraction = 1;

		if(currentVel.x > 0){

			impulse.x = data->jumpImpulse.x * jumpFraction;
		}else{
			impulse.x = -data->jumpImpulse.x * jumpFraction; 
		}
		impulse.y += data->jumpImpulse.y * jumpFraction;
	}*/

	if(impulse.x < 0){
		impulse.x = std::max(impulse.x, -data->jumpImpulse.x);
	}else{
		impulse.x = std::min(impulse.x, data->jumpImpulse.x);
	}
	return impulse;

}
