#include "../defines/Collisions.h"
#include "BulletProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"


void bulletCollisionCallback(CollisionData &collision, void *data){
	Object *obj = collision.me;
	assert(obj != NULL);

	Object *other = collision.otherObj;
	const Hash *otherCollision = collision.otherPhy->collisionType;

	BulletData *bulletData = obj->getPrimitive<BulletData>(Hash::getHash("BulletData"));
	assert(bulletData != NULL);


	//ignore other bullets, and stuff we're instructed to ignore
	if(other->hasProperty(Hash::getHash("BulletData"))) {

		collision.contact->SetEnabled(false);
		return;
	}

	//send the event with the PickupData
	
	//after this, the evenData is deleted
	obj->Kill();
}

void BulletProcessor::_onObjectAdd(Object *obj){
	
	BulletData *data = obj->getPrimitive<BulletData>(Hash::getHash("BulletData"));

	if(data == NULL){
		return;
	};
	
	PhyData *physicsData = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	assert(physicsData != NULL);

	b2Body *body = physicsData->body;
	assert(body != NULL);



	//vector2 g = vector2::cast(world->GetGravity());
	body->ApplyLinearImpulse(body->GetMass() * data->beginVel, body->GetWorldCenter(), true);
	body->SetTransform(body->GetPosition(), data->angle.toRad());

	assert(data->gravityScale >= 0);
	body->SetGravityScale(data->gravityScale);


	for(BulletCollider *collider : data->colliders){
		collider->onCreate(obj);
	}

	for(BulletModifier &modifier : data->modifiers) {
		if(modifier.bulletCreateFunction != NULL) {
			modifier.bulletCreateFunction(obj, modifier.createData);
		}
	}


	CollisionHandler bulletCollisionHandler;
	//if otherCollision is NULL, it lets us collide against ANYTHING
	bulletCollisionHandler.otherCollision = NULL;
	bulletCollisionHandler.onBegin = bulletCollisionCallback;
	bulletCollisionHandler.data = NULL;

	physicsData->collisionHandlers.push_back(bulletCollisionHandler);

	

};

void BulletProcessor::_Process(Object *obj, float dt){


	BulletData *data = obj->getPrimitive<BulletData>(Hash::getHash("BulletData"));
	
	PhyData *physicsData = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	assert(physicsData != NULL);

	b2Body *body = physicsData->body;		

	/*
	for(CollisionData collision : physicsData->collisions){
		this->_handleCollision(collision, data, obj);
	}*/
	

};

/*
void BulletProcessor::_handleCollision(CollisionData &collision, BulletData *data, Object *bullet){

	Object *other = collision.otherObj;
	const Hash *collisionType = collision.getCollidedObjectCollision();
	bool kill = true;


	if(collision.type != CollisionData::Type::onBegin){
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
*/