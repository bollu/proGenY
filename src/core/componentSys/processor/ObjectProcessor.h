#pragma once
#include "../Object.h"
#include "../../IO/logObject.h"
#include <vector>
#include "../../math/mathUtil.h"



/*!Used to process Object classes 

*/ 
class ObjectProcessor{
protected:
	//a pointer to a vector of objects. it's owned by objectManager  
	Object::objectMap *objMap;
	
	//name of the object processor 
	std::string name;

	virtual void _Init(){};

	virtual bool _shouldProcess(Object *obj) = 0;
	virtual void _onObjectAdd(Object *obj){};
	virtual void _onObjectDeath(Object *obj){};
	virtual void _onObjectActivate(Object *obj){};
	virtual void _onObjectDeactivate(Object *obj){};
	virtual void _Process(Object *obj, float dt){};


	bool hasProcessToken(Object *obj){
		return obj->hasProperty(this->name.c_str());
	}

	void addProcessToken(Object *obj){
		obj->addProp(this->name.c_str(), new dummyProp());
	}


	ObjectProcessor(std::string _name) : name(_name){};

public:
	void Init(Object::objectMap *_objMap){
		this->objMap = _objMap;
		
	}

	virtual void onObjectAdd(Object *obj){

		if(_shouldProcess(obj)){
			this->addProcessToken(obj);
			this->_onObjectAdd(obj);
		}
	};
	virtual void onObjectDeath(Object *obj){
		if(hasProcessToken(obj)){
			this->_onObjectDeath(obj);
		}
	};
	
	virtual void onObjectActivate(Object *obj) {
		if(hasProcessToken(obj)){
			this->_onObjectActivate(obj);
		}
	};
	virtual void onObjectDeactivate(Object *obj) {
		if(hasProcessToken(obj)){
			this->_onObjectDeactivate(obj);
		}
	};
	
	virtual void preProcess(){};

	virtual void Process(float dt){
		for(auto it : (*this->objMap)){
			Object *obj = it.second;
			if(_shouldProcess(obj)){
				this->_Process(obj, dt);
			}
		}
	};


	virtual void postProcess(){};
	virtual ~ObjectProcessor(){};

};
