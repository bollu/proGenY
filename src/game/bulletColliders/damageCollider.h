#pragma once
#include "../ObjProcessors/bulletProcessor.h"
#include "../ObjProcessors/healthProcessor.h"

class damageCollider : public bulletCollider{
private:
	float damage;
public:
	damageCollider(float damage){
		this->damage = damage;
	};

	~damageCollider(){};

	bool onEnemyCollision(collisionData &collision, Object *bullet){
		Object *other = collision.otherObj;

		healthData *health = other->getPrimitive<healthData>(Hash::getHash("healthData"));
		if(health == NULL)
			return true;

		health->Damage(this->damage);

		return true;
	}
};
