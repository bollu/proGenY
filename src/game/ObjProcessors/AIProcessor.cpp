#pragma once
#include "AIProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"

void AIProcessor::onObjectAdd(Object *obj){};

void AIProcessor::_Process(Object *obj, float dt){
	AIData *ai = obj->getPrimitive<AIData>(Hash::getHash("AIData"));

	if (ai->target == NULL) { return; }
	
	
	PhyData *phy = obj->getPrimitive<PhyData>(Hash::getHash("PhyData"));
	b2Body *body = phy->body;

	vector2 objPos = *obj->getPrimitive<vector2>(Hash::getHash("position"));
	vector2 targetPos = *ai->target->getPrimitive<vector2>(Hash::getHash("position"));
	
	float k = 1; float q = 1;

	vector2 dist = objPos - targetPos;
	//this is the "x" in F = -kx
	float stretch = dist.Length() - ai->separation;

	vector2 stretchVec = k * dist.Normalize() * stretch;
	vector2 velVec = 1 * vector2::cast(body->GetLinearVelocity());

	vector2 force = -1 * (stretchVec + velVec);
	
	body->ApplyLinearImpulse(ai->speed * force * dt, body->GetWorldCenter(), true);
};
