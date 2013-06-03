#pragma once
#include "Object.h"
#include "../util/logObject.h"
#include <vector>


typedef std::map<std::string, Object *> objectMap;
typedef objectMap::iterator objMapIt;
typedef objectMap::const_iterator cObjMapIt;

/*!Used to process Object classes 

*/ 
class objectProcessor{
protected:
	//a pointer to a vector of objects. it's owned by objectManager  
	  objectMap *objMap;

	  virtual void _Init(){};
public:
	void Init(objectMap *_objMap){
		this->objMap = _objMap;
		
	}

	virtual void onObjectAdd(Object *obj){};
	virtual void onObjectRemove(Object *obj){};
	
	virtual void preProcess(){};
	virtual void Process(float dt) = 0;
	virtual void postProcess(){};

	virtual ~objectProcessor(){};

};