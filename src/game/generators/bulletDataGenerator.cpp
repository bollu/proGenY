#pragma once
#include "bulletDataGenerator.h"

bulletDataGenerator::bulletDataGenerator(unsigned int power, unsigned long seed){
	this->seed = seed;
	this->power = power;

	generator.seed(seed);
};



#pragma once
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../ObjProcessors/bulletProcessor.h"
#include "../bulletColliders/damageCollider.h"
#include "../bulletColliders/pushCollider.h"

void bulletDataGenerator::_createBulletCollider(unsigned long colliderSeed){

};


bulletData bulletDataGenerator::Generate(){
	bulletData data;

	const int numChoices = 3;
	
	int rand = generator() % numChoices;

	switch(rand){
		case 0:
			
		case 1:
		case 2:
			break;
	};


	return data;
};