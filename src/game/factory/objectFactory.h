#pragma once
#include "../../core/Object.h"


class objectCreator{
protected:
	objectCreator(){}
	//make sure no once can destroy this -_-
	virtual ~objectCreator(){};
	friend class objectFactory;
public:
};

class objectFactory{
private:
	
	std::map<const Hash *, objectCreator* >creators; 	
public:
	template <class T>
	T *getCreator(const Hash* objName){
		auto it = this->creators.find(objName);

		assert(it != this->creators.end());

		T *creator = dynamic_cast<T *>((it->second));
		assert(creator != NULL);
		return creator;
	};


	//!this will keep a copy of the pointer, so DO NOT
	//free the objectCreator once this function has been called
	void attachObjectCreator(const Hash *objName, objectCreator *creator);
};
