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
	
	float k = ai->separationCoeff; float q = ai->separationCoeff;

	vector2 dist = objPos - targetPos;
	float distLength = dist.Length();

	if (distLength > ai->maxLockonRange) { return; }
	//this is the "x" in F = -kx
	float stretch = distLength - ai->separation;

	vector2 stretchVec = k * dist.Normalize() * stretch;
	vector2 velVec = 1 * vector2::cast(body->GetLinearVelocity());

	vector2 force = -1 * (stretchVec + velVec);
	
	body->ApplyLinearImpulse(ai->forceScaling * force * dt, body->GetWorldCenter(), true);
};
