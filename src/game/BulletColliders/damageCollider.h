#pragma once
#include "../ObjProcessors/StabProcessor.h"
#include "../ObjProcessors/HealthProcessor.h"

class damageCollider : public StabCollider{
private:
	float damage;
public:
	damageCollider(float damage){
		this->damage = damage;
	};

	~damageCollider(){};

	bool onEnemyCollision(CollisionData &collision, Object *bullet){
		Object *other = collision.otherObj;
		other->sendMessage<int>(Hash::getHash("DamageHP"), this->damage);
		return true;
	}
};
