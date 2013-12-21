
#include "objectFactory.h"



void objectFactory::attachObjectCreator(const Hash *objName, objectCreator *creator){

	auto it = this->creators.find(objName);
	assert(it == this->creators.end());

	this->creators[objName] = creator;
};
