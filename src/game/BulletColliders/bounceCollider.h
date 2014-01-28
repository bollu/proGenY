#pragma once
#include "../ObjProcessors/StabProcessor.h"
#include "../ObjProcessors/HealthProcessor.h"

class bounceCollider : public StabCollider{
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

	bool onEnemyCollision(CollisionData &collision, Object *bullet){
		
		b2Manifold *manifold = collision.contact->GetManifold();
		
		b2Body *myBody = collision.myPhy->body;

		vector2 normal = vector2::cast(manifold->localNormal);
		vector2 myApproachVel = vector2::cast(collision.otherPhy->body->GetLinearVelocity() - 
			myBody->GetLinearVelocity());

		vector2 velAlongNormal = myApproachVel.projectOn(normal);
		vector2 velAlongTangent = myApproachVel - velAlongNormal;

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

	bool onDefaultCollision(CollisionData &collision, Object *bullet){
		 return  true;

	}
};
