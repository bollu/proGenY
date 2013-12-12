#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"

#include "../../core/Process/viewProcess.h"
#include "../../core/Process/worldProcess.h"
#include "../../util/Cooldown.h"
#include <unordered_set>

struct collisionData;


class BulletCollider{
protected:
	BulletCollider(){};
public:
	virtual ~BulletCollider(){};

	virtual void onCreate(Object *bullet){};
	/*!handle the collision with an enemy
	*return whether the bullet should be killed
	*/
	virtual bool onEnemyCollision(collisionData &data, Object *bullet) = 0;
	/*! handle the collision with any other object type that is not ignored 
	*return whether the bullet should be killed*/
	virtual bool onDefaultCollision(collisionData &data, Object *bullet){
		return true;
	}	

	virtual void onDeath(collisionData &data, Object *bullet){};

};


struct BulletData{
private:
	friend class bulletProcessor;

	//!accelerate the bullet after some time
	float latentAccel;
	Cooldown<float> latentAccelCooldown;

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
		latentAccel = 0.0;
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

	void addLatentAcceleration(float startTime, float acceleration) {
		this->latentAccel = acceleration;
		this->latentAccelCooldown.setTotalTime(startTime);
		this->latentAccelCooldown.startCooldown();
	}
};


class bulletProcessor : public objectProcessor{
public:
	bulletProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager){
		this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));
	}

	void onObjectAdd(Object *obj);
	void Process(float dt);


private:
	void _handleCollision(collisionData &collision,BulletData *data, Object *obj);
	worldProcess *world;

};