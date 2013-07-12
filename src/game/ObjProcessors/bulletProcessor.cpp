#pragma once
#include "bulletProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"


void bulletProcessor::onObjectAdd(Object *obj){
	
	bulletData *data = obj->getProp<bulletData>(Hash::getHash("bulletData"));

	if(data == NULL){
		return;
	};
	
	phyData *physicsData = obj->getProp<phyData>(Hash::getHash("phyData"));
	assert(physicsData != NULL);

	b2Body *body = physicsData->body;
	assert(body != NULL);

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

		/*
		vector2 vel = vector2::cast(body->GetLinearVelocity());
		body->SetTransform(body->GetPosition(), vel.toAngle());
		*/
		
		for(collisionData collision : physicsData->collisions){
			this->_handleCollision(collision, data, obj);
		}
	};

};

void bulletProcessor::_handleCollision(collisionData &collision, bulletData *data, Object *bullet){

	Object *other = collision.obj;
	const Hash *collisionType = collision.phy->collisionType;
	bool kill = true;

	if(other->hasProperty(Hash::getHash("bulletData"))){
		return;
	}

	/*for(const Hash* ignore : data->ignoreCollisions){
		if(collisionType == ignore){
			kill = false;
			return;
		}
	}*/

	if(data->ignoreCollisions.count(collisionType) > 0){
		kill = false;
		return;
	}


	if(data->enemyCollisions.count(collisionType) > 0){
		kill = true;

		for(bulletCollider *collider : data->colliders){
			//only if ALL bullet colliders agree, kill the bullet
			kill = kill && collider->onCollision(collision, bullet);
		}

	}
	/*for(const Hash *enemy : data->enemyCollisions){

		kill = true;

		if(collisionType == enemy){

			for(bulletCollider *collider : data->colliders){
				//only if ALL bullet colliders agree, kill the bullet
				kill = kill && collider->onCollision(collision, bullet);
			}
		}
	}*/
	

	if(kill){
		bullet->Kill();
	};
};
