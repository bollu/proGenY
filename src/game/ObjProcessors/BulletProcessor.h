#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/math/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"

#include "../../core/Rendering/viewProcess.h"
#include "../../core/World/worldProcess.h"

#include <unordered_set>

struct CollisionData;

typedef bool (*onEnemyCollision)(CollisionData&, Object*);
typedef void (*onDefaultCollision)(Object*);
typedef void (*onBulletCreate)(Object*);

class BulletCollider{
protected:
	BulletCollider(){};
public:
	virtual ~BulletCollider(){};

	virtual void onCreate(Object *bullet){};
	/*!handle the collision with an enemy
	*return whether the bullet should be killed
	*/
	virtual bool onEnemyCollision(CollisionData &data, Object *bullet) = 0;
	/*! handle the collision with any other object type that is not ignored 
	*return whether the bullet should be killed*/
	virtual bool onDefaultCollision(CollisionData &data, Object *bullet){
		return true;
	}	

	virtual void onDeath(CollisionData &data, Object *bullet){};

};


struct BulletData{
public:
	vector2 beginVel;
	/*!Angle to face in the beginning in degrees*/
	util::Angle angle;

	//!amount by which gravity affects the bullet
	float gravityScale;

	//! BulletColliders that handle what happens during collision
	std::vector<BulletCollider *> colliders;

	//collision types considered to be enemies
	std::unordered_set<const Hash*> enemyCollisions;
	//collision types to be ignored
	std::unordered_set<const Hash*> ignoreCollisions;
		
	BulletData(){
		gravityScale = 3.0;
	};

	void addEnemyCollision(const Hash *collision){
		this->enemyCollisions.insert(collision);
	}

	void addIgnoreCollision(const Hash *collision){
		this->ignoreCollisions.insert(collision);
	}

	void addBulletCollder(BulletCollider *collider){
		this->colliders.push_back(collider);
	}
};


class BulletProcessor : public ObjectProcessor{
public:
	BulletProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
		ObjectProcessor("BulletProcessor"){

			this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));
	}


protected:
	void _onObjectAdd(Object *obj);
	void _Process(Object *obj, float dt);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("BulletData") && obj->requireProperty("PhyData");
	};

private:
	void _handleCollision(CollisionData &collision,BulletData *data, Object *obj);
	worldProcess *world;

};
