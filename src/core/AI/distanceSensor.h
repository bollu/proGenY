#pragma once
#include "baseAI.h"
class b2World;


class distanceSensor : public AI::Sensor{
protected:
	friend class Task;
	std::string targetClass;

public:
	distanceSensor(Object *owner, b2World *world, std::string targetClass);

	void Sense();
};
