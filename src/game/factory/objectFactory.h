#pragma once
#include "../../core/Object.h"


class objectCreator{
protected:
	//make sure no once can destroy this -_-
	~objectCreator(){};
	friend class objectFactory;
public:
	virtual Object *createObject(vector2 pos) const= 0;
};

class objectFactory{
private:
	
	std::map<const Hash *, objectCreator* >creators; 	
public:

	
	objectCreator *getCreator(const Hash* objName);

	//!this will keep a copy of the pointer, so DO NOT
	//free the objectCreator once this function has been called
	void attachObjectCreator(const Hash *objName, objectCreator *creator);
};
