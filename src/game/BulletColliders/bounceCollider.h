#pragma once
#include "../ObjProcessors/BulletProcessor.h"
#include "../ObjProcessors/HealthProcessor.h"

class bounceCollider : public BulletCollider{
private:
	int totalBounces;

public:

	bounceCollider(unsigned int numBounces){
		this->totalBounces = numBounces;
	};

	void onCreate(Object *bullet){
		bullet->addProp(Hash::getHash("bulletNumBounces"), new Prop<int>(totalBounces));
	}

	~bounceCollider(){};

	bool onEnemyCollision(collisionData &collision, Object *bullet){

		
		//vector2 normal = ->//collision.normal;
		b2Body *myBody = collision.myPhy->body;

		vector2 myVel = collision.myApproachVel;
		//vector2::cast(myBody->GetLinearVelocity());
		
		vector2 normal = collision.normal;
		vector2 velAlongNormal = myVel.projectOn(normal);
		vector2 velAlongTangent = myVel - velAlongNormal;

		vector2 resultant = -1 * velAlongNormal +  velAlongTangent;
		//apply the impulse
		myBody->SetLinearVelocity(zeroVector);
		myBody->ApplyLinearImpulse(resultant, myBody->GetWorldCenter(), true);

		if(!bullet->hasProperty("bulletNumBounces")){
			IO::errorLog<<"\n\nno property on bullet"<<IO::flush;
		}

		int *numBounces = bullet->getPrimitive<int>(Hash::getHash("bulletNumBounces"));
		*numBounces = (*numBounces  - 1);

		return (*numBounces < 0);
	};

	bool onDefaultCollision(collisionData &collision, Object *bullet){
		 return  this->onEnemyCollision(collision, bullet);

	}
};
