#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/math/vector.h"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"

#include "../../core/Rendering/viewProcess.h"
#include "../../core/World/worldProcess.h"

#include <unordered_set>

struct CollisionData;
/*
typedef bool (*onEnemyCollision)(CollisionData& collision, Object* bullet, void* data);
typedef void (*onDefaultCollision)(Object* bullet, void *data);
typedef void (*onBulletCreate)(Object* bullet, void *data);
typedef void (*onBulletThink)(Object* bullet, void *data);

struct DamageModifier {
	onEnemyCollision enemyCollisionFunc = NULL;
	onDefaultCollision defaultCollisionFunc = NULL;
	onBulletCreate bulletCreateFunction = NULL;
	onBulletThink bulletThinkFunction = NULL;

	void *enemyCollisionData, *defaultCollisionData, *createData, *thinkData;

};
*/

//class that represents what should happen on collision
class StabCollider{
protected:
	StabCollider(){};
public:
	virtual ~StabCollider(){};

	virtual void onCreate(Object *bullet){};
	/*!handle the collision with an enemy
	*return whether the bullet should be killed
	*/
	virtual bool onEnemyCollision(CollisionData &data, Object *stabber) = 0;
	/*! handle the collision with any other object type that is not ignored 
	*return whether the bullet should be killed*/
	virtual bool onDefaultCollision(CollisionData &data, Object *stabber){
		return true;
	}	

	virtual void onDeath(CollisionData &data, Object *stabber){};

};


struct StabData { 
	//whether the object should be killed on colision by deafult. Note: stabColliders can ovveride this
	bool killOnHit;

	std::vector<StabCollider *> colliders;

	//collision types considered to be enemies
	std::unordered_set<const Hash*> enemyCollisions;


	void addEnemyCollision(const Hash *collision){
		this->enemyCollisions.insert(collision);
	}

	void addCollider(StabCollider *collider){
		this->colliders.push_back(collider);
	}
};

class StabProcessor : public ObjectProcessor {
public:
	StabProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
		ObjectProcessor("StabProcessor"){

			this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));
	}

protected:
	void _onObjectAdd(Object *obj);
	
	bool _shouldProcess(Object *obj){
		return obj->hasProperty("StabData") && obj->requireProperty("PhyData");
	};

private:
	//void _handleCollision(CollisionData &collision,BulletData *data, Object *obj);
	worldProcess *world;
};



