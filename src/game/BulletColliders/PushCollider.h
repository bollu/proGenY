#pragma once
#include "../ObjProcessors/StabProcessor.h"



class pushCollider : public StabCollider{
private:
	float impulseMagnitude;
public:
	pushCollider(float impulseMagnitude){
		this->impulseMagnitude = impulseMagnitude;
	};

	~pushCollider(){};

	bool onEnemyCollision(CollisionData &collision, Object *bullet){

		PhyData *other = collision.otherPhy;
		PhyData *bulletPhy = bullet->getPrimitive<PhyData>(Hash::getHash("PhyData"));

		vector2 vel = vector2::cast(bulletPhy->body->GetLinearVelocity());

		vector2 impulse = vel.Normalize() * this->impulseMagnitude;

		other->body->ApplyLinearImpulse(impulse, other->body->GetWorldCenter(), true);

		return true;
	};
};
