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

class bulletCollider
{
protected:
	bulletCollider (){}


public:
	virtual ~bulletCollider (){}

	virtual void onCreate ( Object *bullet ){}

	/*!handle the collision with an enemy
	 * return whether the bullet should be killed
	 */
	virtual bool onEnemyCollision ( collisionData &data, Object *bullet ) = 0;

	/*! handle the collision with any other object type that is not ignored
	 * return whether the bullet should be killed*/
	virtual bool onDefaultCollision ( collisionData &data, Object *bullet ){
		return (true);
	}


	virtual void onDeath ( collisionData &data, Object *bullet ){}
};


class bulletProp : public baseProperty{
public:

	vector2 beginVel;

	/*!Angle to face in the beginning in degrees*/
	util::Angle angle;

	//!amount by which gravity affects the bullet
	float gravityScale;

	//! bulletColliders that handle what happens during collision
	std::vector< bulletCollider * > colliders;

	//collision types considered to be enemies
	std::unordered_set< const Hash * > enemyCollisions;

	//collision types to be ignored
	std::unordered_set< const Hash * > ignoreCollisions;
	bulletProp (){
		gravityScale = 3.0;
	}


	bulletProp (const bulletProp &other){
		this->beginVel = other.beginVel;
		this->angle = other.angle;
		this->gravityScale = other.gravityScale;
		
		this->colliders = other.colliders;
		this->enemyCollisions = other.enemyCollisions;
		this->ignoreCollisions = other.ignoreCollisions;
	}


	void operator = (const bulletProp &other){
		this->beginVel = other.beginVel;
		this->angle = other.angle;
		this->gravityScale = other.gravityScale;
		
		this->colliders = other.colliders;
		this->enemyCollisions = other.enemyCollisions;
		this->ignoreCollisions = other.ignoreCollisions;
	}

	
	void addEnemyCollision ( const Hash *collision ){
		this->enemyCollisions.insert( collision );
	}


	void addIgnoreCollision ( const Hash *collision ){
		this->ignoreCollisions.insert( collision );
	}


	void addBulletCollder ( bulletCollider *collider ){
		this->colliders.push_back( collider );
	}
};



class bulletProcessor : public objectProcessor
{
public:
	bulletProcessor ( processMgr &processManager, Settings &settings, eventMgr &_eventManager );
	void onObjectAdd ( Object *obj );
	void Process ( float dt );


private:
	void _handleCollision ( collisionData &collision, bulletProp *data, Object *obj );

	b2World *world;
};