#pragma once
#include "bulletProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "healthProcessor.h"

void bulletProcessor::onObjectAdd(Object *obj){
	
	bulletData *data = obj->getProp<bulletData>(Hash::getHash("bulletData"));

	if(data == NULL){
		return;
	};
	

	util::msgLog(obj->getName() + " has bullet");
	
	phyData *physicsData = obj->getProp<phyData>(Hash::getHash("phyData"));
	assert(physicsData != NULL);

	b2Body *body = physicsData->body;
	assert(body != NULL);

	PRINTVECTOR2(data->beginVel);

	vector2 g = vector2::cast(world->GetGravity());
	body->ApplyLinearImpulse(body->GetMass() * data->beginVel, body->GetWorldCenter());
	body->SetTransform(body->GetPosition(), data->angle.toRad());
	

};

void bulletProcessor::Process(float dt){

	for(auto it=  objMap->begin(); it != objMap->end(); ++it){


		Object *obj = it->second;

		bulletData *data = obj->getProp<bulletData>(Hash::getHash("bulletData"));

	
		if(data == NULL){
			continue;
		}

		
		phyData *physicsData = obj->getProp<phyData>(Hash::getHash("phyData"));
		assert(physicsData != NULL);

		b2Body *body = physicsData->body;

		vector2 vel = vector2::cast(body->GetLinearVelocity());
		body->SetTransform(body->GetPosition(), vel.toAngle());

		
		for(collisionData collision : physicsData->collisions){

			
			if(collision.data->collisionType == data->enemyCollision){
				//TODO: implement damage, not really sure how to go about this
				//this solution is good enough for now
				
				healthData *health = collision.obj->getProp<healthData>(Hash::getHash("healthData"));

				if(health == NULL){
					return;
				}
				
				health->Damage(data->damage);

			}
			else{
				obj->Kill();
			}
		
		}
	};

};
