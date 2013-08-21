#pragma once
#include "../ObjProcessors/bulletProcessor.h"


class pushCollider : public bulletCollider
{
private:
	float impulseMagnitude;


public:
	pushCollider ( float impulseMagnitude ){
		this->impulseMagnitude = impulseMagnitude;
	}


	~pushCollider (){}

	bool onEnemyCollision ( collisionData &collision, Object *bullet ){
		
		phyProp *other	= collision.otherPhy;
		phyProp *bulletPhy = bullet->getComplexProp<phyProp>(
							 Hash::getHash( "phyProp" ));

		assert(other != NULL);
		assert(bulletPhy != NULL);

		vector2 vel	= vector2::cast( bulletPhy->body->GetLinearVelocity() );
		vector2 impulse	= vel.Normalize() * this->impulseMagnitude;


		other->body->ApplyLinearImpulse( impulse, 
			other->body->GetWorldCenter() );

		return (true);
	} //onEnemyCollision
};