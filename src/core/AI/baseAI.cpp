#pragma once
#include "baseAI.h"

using namespace AI;

//--------------------------------------------------------------------------------
//sensor--------------------------------------------------------------------------
Sensor::Sensor(Object *_owner,std::string _name) : owner(_owner), name(_name){}

std::string Sensor::getName() const{
	return this->name;
}
//--------------------------------------------------------------------------------
//task----------------------------------------------------------------------------
Task::Task(Object::objectMap &_objectMap, Object *_owner) : objectMap(_objectMap), owner(_owner){};

bool Task::isExclusive() const{
	return false;
};

//--------------------------------------------------------------------------------
//executor------------------------------------------------------------------------
Executor::Executor(Object::objectMap &_objectMap, Object *_owner) : objectMap(_objectMap), owner(_owner){};
