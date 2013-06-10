#pragma once
#include "objectFactory.h"


objectCreator *objectFactory::getCreator(const Hash* objName){
	auto it = this->creators.find(objName);;

	assert(it != this->creators.end());

	objectCreator *creator = it->second;
	return creator;
	
};

void objectFactory::attachObjectCreator(const Hash *objName, objectCreator *creator){

	auto it = this->creators.find(objName);
	assert(it == this->creators.end());

	this->creators[objName] = creator;
};
