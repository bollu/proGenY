#pragma once
#include "../ObjProcessors/bulletProcessor.h"
#include "../ObjProcessors/healthProcessor.h"

class bounceCollider : public bulletCollider{
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

		vector2 myVel = vector2::cast(myBody->GetLinearVelocity());
		
		vector2 normal = collision.normal;
		vector2 velAlongNormal = myVel.projectOn(normal);
		vector2 velAlongTangent = myVel - velAlongNormal;

		vector2 resultant = -1 * velAlongNormal +  velAlongTangent;
		//apply the impulse
		myBody->SetLinearVelocity(zeroVector);
		myBody->ApplyLinearImpulse(resultant, myBody->GetWorldCenter());

		util::infoLog("------------");
		PRINTVECTOR2(normal);
		PRINTVECTOR2(myVel);
		PRINTVECTOR2(velAlongNormal);
		PRINTVECTOR2(velAlongTangent);
		PRINTVECTOR2(resultant);

		Prop<int> *numBouncesProp = bullet->getPropPtr<int>(Hash::getHash("bulletNumBounces"));
		int numBounces = *numBouncesProp->getVal();
		numBouncesProp->setValStack(numBounces  - 1);

		return (numBounces <= 0);
	};

	bool onDefaultCollision(collisionData &collision, Object *bullet){
		 return  this->onEnemyCollision(collision, bullet);

	}
};