#pragma once
#include "../ObjProcessors/BulletProcessor.h"



class pushCollider : public BulletCollider{
private:
	float impulseMagnitude;
public:
	pushCollider(float impulseMagnitude){
		this->impulseMagnitude = impulseMagnitude;
	};

	~pushCollider(){};

	bool onEnemyCollision(collisionData &collision, Object *bullet){

		PhyData *other = collision.otherPhy;
		PhyData *bulletPhy = bullet->getPrimitive<PhyData>(Hash::getHash("PhyData"));

		vector2 vel = vector2::cast(bulletPhy->body->GetLinearVelocity());

		vector2 impulse = vel.Normalize() * this->impulseMagnitude;

		other->body->ApplyLinearImpulse(impulse, other->body->GetWorldCenter(), true);

		return true;
	};
};
