#pragma once
#include "bulletProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"


void bulletProcessor::onObjectAdd(Object *obj){
	
	BulletData *data = obj->getProp<BulletData>(Hash::getHash("BulletData"));

	if(data == NULL){
		return;
	};
	
	phyData *physicsData = obj->getProp<phyData>(Hash::getHash("phyData"));
	assert(physicsData != NULL);

	b2Body *body = physicsData->body;
	assert(body != NULL);


	body->ApplyLinearImpulse(body->GetMass() * data->beginVel, body->GetWorldCenter(), true);
	body->SetTransform(body->GetPosition(), data->angle.toRad());

	assert(data->gravityScale >= 0);
	body->SetGravityScale(data->gravityScale);


	for(BulletCollider *collider : data->colliders){
		collider->onCreate(obj);
	}

};

void bulletProcessor::Process(float dt){

	for(auto it =  objMap->begin(); it != objMap->end(); ++it){

		Object *obj = it->second;

		BulletData *data = obj->getProp<BulletData>(Hash::getHash("BulletData"));
		if(data == NULL){
			continue;
		}	

		phyData *physicsData = obj->getProp<phyData>(Hash::getHash("phyData"));
		assert(physicsData != NULL);

		b2Body *body = physicsData->body;	

		if (data->latentAccel != 0.0 && data->latentAccelCooldown.onCooldown()){
			//it switch from onCooldown to offCooldown
			if(data->latentAccelCooldown.Tick(dt).offCooldown()){
				
				body->ApplyLinearImpulse(body->GetMass() * data->angle.polarProjection() * data->latentAccel, body->GetWorldCenter(), true);
				//body->ApplyForceToCenter(data->angle.polarProjection() * data->latentAccel * body->GetMass(), true);
			}

		}

		for(collisionData collision : physicsData->collisions){
			this->_handleCollision(collision, data, obj);
		}
	};

};

void bulletProcessor::_handleCollision(collisionData &collision, BulletData *data, Object *bullet){

	Object *other = collision.otherObj;
	const Hash *collisionType = collision.getCollidedObjectCollision();
	bool kill = true;


	if(collision.type != collisionData::Type::onBegin){
		return;
	}
	//ignore other bullets
	if(other->hasProperty(Hash::getHash("BulletData"))){
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

		for(BulletCollider *collider : data->colliders){
			//only if ALL bullet colliders agree, kill the bullet
			kill = kill && collider->onEnemyCollision(collision, bullet);
		}

	}
	else{


		kill = true;
		for(BulletCollider *collider : data->colliders){
				//only if ALL bullet colliders agree, kill the bullet
			kill = kill && collider->onDefaultCollision(collision, bullet);
		}
		
	}

	
	if(kill){
		bullet->Kill();
	};
};
