#include "../defines/Collisions.h"
#include "StabProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"


bool isEnemy(std::unordered_set<const Hash*> &enemyCollisions, const Hash *objCollisionType) {
	return enemyCollisions.find(objCollisionType) != enemyCollisions.end();
}
void stabCollisionCallback(CollisionData &collision, void *data){
	
	Object *obj = collision.me;
	assert(obj != NULL);

	Object *other = collision.otherObj;
	const Hash *otherCollision = collision.otherPhy->collisionType;

	StabData *stabData = obj->getPrimitive<StabData>(Hash::getHash("StabData"));
	assert(stabData != NULL);

	bool kill = stabData->killOnHit;

	//if the collision is an enemy
	if (isEnemy(stabData->enemyCollisions, otherCollision)) {

		for(StabCollider *collider : stabData->colliders){
			kill = kill && collider->onEnemyCollision(collision, obj);
		}
	}
	else {
		for(StabCollider *collider : stabData->colliders){
			kill = kill && collider->onDefaultCollision(collision, obj);
		}
	}

	
	if (kill) {
		obj->Kill();
	}
}

void StabProcessor::_onObjectAdd(Object *obj){
	
	StabData *data = obj->getPrimitive<StabData>(Hash::getHash("StabData"));

	PhyData *physicsData = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	assert(physicsData != NULL);
	
	for(StabCollider *collider : data->colliders){
		collider->onCreate(obj);
	}

	
	CollisionHandler stabCollisionHandler;
	//if otherCollision is NULL, it lets us collide against ANYTHING
	stabCollisionHandler.otherCollision = CollisionHandler::ALL_COLLISIONS;
	stabCollisionHandler.onBegin = stabCollisionCallback;
	stabCollisionHandler.data = NULL;

	physicsData->collisionHandlers.push_back(stabCollisionHandler);

	

};
