#pragma once
#include "bulletProcessor.h"
#include "../../core/componentSys/processor/phyProcessor.h"


void bulletProcessor::_onObjectAdd(Object *obj){
	
	bulletData *data = obj->getPrimitive<bulletData>(Hash::getHash("bulletData"));

	if(data == NULL){
		return;
	};
	
	phyData *physicsData = obj->getPrimitive<phyData>(Hash::getHash("phyData"));
	assert(physicsData != NULL);

	b2Body *body = physicsData->body;
	assert(body != NULL);

	//vector2 g = vector2::cast(world->GetGravity());
	body->ApplyLinearImpulse(body->GetMass() * data->beginVel, body->GetWorldCenter(), true);
	body->SetTransform(body->GetPosition(), data->angle.toRad());

	assert(data->gravityScale >= 0);
	body->SetGravityScale(data->gravityScale);


	for(bulletCollider *collider : data->colliders){

		collider->onCreate(obj);
	}


	

};

void bulletProcessor::_Process(Object *obj, float dt){


	bulletData *data = obj->getPrimitive<bulletData>(Hash::getHash("bulletData"));
	
	phyData *physicsData = obj->getPrimitive<phyData>(Hash::getHash("phyData"));
	assert(physicsData != NULL);

	b2Body *body = physicsData->body;		
	for(collisionData collision : physicsData->collisions){
		this->_handleCollision(collision, data, obj);
	}
	

};

void bulletProcessor::_handleCollision(collisionData &collision, bulletData *data, Object *bullet){

	Object *other = collision.otherObj;
	const Hash *collisionType = collision.getCollidedObjectCollision();
	bool kill = true;


	if(collision.type != collisionData::Type::onBegin){
		return;
	}
	//ignore other bullets
	if(other->hasProperty(Hash::getHash("bulletData"))){
		return;
	}


	//if the collisionType is to be ignored, return
	if(data->ignoreCollisions.count(collisionType) > 0){
		kill = false;
		return;
	}


	//if the collisionType is an enemy, proceed
	if(data->enemyCollisions.count(collisionType) > 0){
		kill = true;

		for(bulletCollider *collider : data->colliders){
			//only if ALL bullet colliders agree, kill the bullet
			kill = kill && collider->onEnemyCollision(collision, bullet);
		}

	}
	else{


		kill = true;
		for(bulletCollider *collider : data->colliders){
				//only if ALL bullet colliders agree, kill the bullet
			kill = kill && collider->onDefaultCollision(collision, bullet);
		}
		
	}

	if(kill){
		bullet->Kill();
	};
};
