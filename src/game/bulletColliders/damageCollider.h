#pragma once
#include "../ObjProcessors/bulletProcessor.h"
#include "../ObjProcessors/healthProcessor.h"

class damageCollider : public BulletCollider{
private:
	float damage;
public:
	damageCollider(float damage){
		this->damage = damage;
	};

	~damageCollider(){};

	bool onEnemyCollision(collisionData &collision, Object *bullet){
		Object *other = collision.otherObj;

		other->sendMessage<float>(Hash::getHash("damageHealth"), this->damage);
		
		/*
		HealthData *health = other->getProp<HealthData>(Hash::getHash("HealthData"));
		if(health == NULL)
			return true;

		health->Damage(this->damage);
		*/

		return true;
	}
};
