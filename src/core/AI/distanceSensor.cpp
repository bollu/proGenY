#pragma once
#include "distanceSensor.h"

using namespace AI;


distanceSensor::distanceSensor(Object::objectMap  &objectMap, Object *owner, 
	std::string _targetClass) : Sensor(objectMap, owner, "distanceSensor"), 
								targetClass(_targetClass){};


void distanceSensor::Sense(){
	
};
