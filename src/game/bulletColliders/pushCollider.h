#pragma once
#include "../ObjProcessors/bulletProcessor.h"



class pushCollider : public BulletCollider{
private:
	float impulseMagnitude;
public:
	pushCollider(float impulseMagnitude){
		this->impulseMagnitude = impulseMagnitude;
	};

	~pushCollider(){};

	bool onEnemyCollision(collisionData &collision, Object *bullet){

		phyData *other = collision.otherPhy;
		phyData *bulletPhy = bullet->getProp<phyData>(Hash::getHash("phyData"));

		vector2 vel = vector2::cast(bulletPhy->body->GetLinearVelocity());

		vector2 impulse = vel.Normalize() * this->impulseMagnitude;

		other->body->ApplyLinearImpulse(impulse, other->body->GetWorldCenter(), true);

		return true;
	};
};