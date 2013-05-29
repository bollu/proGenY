#pragma once
#include "Object.h"
#include "../util/logObject.h"
#include <vector>


typedef std::map<std::string, Object *> objectMap;
typedef objectMap::iterator objMapIt;
typedef objectMap::const_iterator cObjMapIt;

class objectProcessor{
protected:
	//a pointer to a vector of objects. it's owned by objectManager  
	  const objectMap *objMap;

	  virtual void _Init(){};
public:
	void Init(const objectMap *_objMap){
		this->objMap = _objMap;
		
	}

	virtual void onObjectAdd(Object *obj){};
	virtual void onObjectRemove(Object *obj){};
	
	virtual void Process() = 0;

	virtual ~objectProcessor(){};

};