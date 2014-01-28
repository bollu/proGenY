#include "../defines/Collisions.h"
#include "BulletProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"


void BulletProcessor::_onObjectAdd(Object *obj){

	BulletData *data = obj->getPrimitive<BulletData>(Hash::getHash("BulletData"));
	
	PhyData *physicsData = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	assert(physicsData != NULL);

	b2Body *body = physicsData->body;
	assert(body != NULL);



	//vector2 g = vector2::cast(world->GetGravity());
	body->ApplyLinearImpulse(body->GetMass() * data->beginVel, body->GetWorldCenter(), true);
	body->SetTransform(body->GetPosition(), data->angle.toRad());

	assert(data->gravityScale >= 0);
	body->SetGravityScale(data->gravityScale);
};