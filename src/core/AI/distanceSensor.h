#pragma once
#include "baseAI.h"



class distanceSensor : public AI::Sensor{
protected:
	friend class Task;
	std::string targetClass;

public:
	distanceSensor(Object::objectMap  &objectMap, Object *owner, std::string targetClass);

	void Sense();
};
