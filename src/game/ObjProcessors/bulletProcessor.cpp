#pragma once
#include "bulletProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"



bulletProcessor::bulletProcessor ( processMgr &processManager, 
	Settings &settings, eventMgr &_eventManager ){
	this->world = 
	processManager.getProcess<worldProcess>(Hash::getHash( "worldProcess" ))->getWorld();
}

void bulletProcessor::onObjectAdd(Object *obj){
	
	bulletProp *data = obj->getComplexProp<bulletProp>(Hash::getHash("bulletProp"));

	if(data == NULL){
		return;
	};
	
	phyProp *physicsData = obj->getComplexProp<phyProp>(Hash::getHash("phyProp"));
	assert(physicsData != NULL);

	b2Body *body = physicsData->body;
	assert(body != NULL);

	//vector2 g = vector2::cast(world->GetGravity());
	body->ApplyLinearImpulse(body->GetMass() * data->beginVel, body->GetWorldCenter());
	body->SetTransform(body->GetPosition(), data->angle.toRad());

	assert(data->gravityScale >= 0);
	body->SetGravityScale(data->gravityScale);

	for(bulletCollider *collider : data->colliders){
		collider->onCreate(obj);
	}	
};

void bulletProcessor::Process(float dt){

	for(auto it=  objMap->begin(); it != objMap->end(); ++it){

		Object *obj = it->second;

		bulletProp *data = obj->getComplexProp<bulletProp>(Hash::getHash("bulletProp"));
		if(data == NULL){
			continue;
		}

		
		phyProp *physicsData = obj->getComplexProp<phyProp>(Hash::getHash("phyProp"));
		assert(physicsData != NULL);

		b2Body *body = physicsData->body;	
		assert(body != NULL);
			
		for(collisionData collision : physicsData->collisions){
			this->_handleCollision(collision, data, obj);
		}
	};

};

void bulletProcessor::_handleCollision(collisionData &collision, bulletProp *data, Object *bullet){

	Object *other = collision.otherObj;
	const Hash *collisionType = collision.getCollidedObjectCollision();
	bool kill = true;


	if(collision.type != collisionData::Type::onBegin){
		return;
	}
	//ignore other bullets
	if(other->hasProperty(Hash::getHash("bulletProp"))){
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
		util::infoLog<<"killing bullet";
		bullet->Kill();
	};
};
