#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"

#include "../../core/Process/viewProcess.h"
#include "../../core/Process/worldProcess.h"

#include <unordered_set>

struct collisionData;


class bulletCollider{
protected:
	bulletCollider(){};
public:
	virtual ~bulletCollider(){};

	virtual void onCreate(Object *bullet){};
	//return whether the bullet should be killed
	virtual bool onCollision(collisionData &data, Object *bullet) = 0;
	virtual void onDeath(collisionData &data, Object *bullet){};

};


struct bulletData{
public:
	vector2 beginVel;
	/*!Angle to face in the beginning in degrees*/
	util::Angle angle;

	//! bulletColliders that handle what happens during collision
	std::vector<bulletCollider *> colliders;

	//collision types considered to be enemies
	std::unordered_set<const Hash*> enemyCollisions;
	//collision types to be ignored
	std::unordered_set<const Hash*> ignoreCollisions;
		
	bulletData(){};

	void addEnemyCollision(const Hash *collision){
		this->enemyCollisions.insert(collision);
	}

	void addIgnoreCollision(const Hash *collision){
		this->ignoreCollisions.insert(collision);
	}

	void addBulletCollder(bulletCollider *collider){
		this->colliders.push_back(collider);
	}
};


class bulletProcessor : public objectProcessor{
public:
	bulletProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager){
		this->world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"))->getWorld();
	}

	void onObjectAdd(Object *obj);
	void Process(float dt);


private:
	void _handleCollision(collisionData &collision,bulletData *data, Object *obj);
	b2World *world;

};