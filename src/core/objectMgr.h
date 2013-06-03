#pragma once
#include "Object.h"
#include "objectProcessor.h"

#include <vector>
#include "../util/logObject.h"


class objectMgr{

private:
	
	objectMap objMap;

	std::vector<objectProcessor *> objProcessors;
	typedef std::vector<objectProcessor *>::iterator objProcessorIt;
public:
	objectMgr(){};
	~objectMgr(){};


	void addObject(Object *obj){
		this->objMap[obj->getName()] = obj;

		for(objProcessorIt it = this->objProcessors.begin(); it != this->objProcessors.end(); ++it){

			(*it)->onObjectAdd(obj);
		}

	}


	void removeObject(std::string name){
		objMapIt it = this->objMap.find(name);

		if(it != this->objMap.end()){
			this->objMap.erase(it);
			Object *obj = it->second;

			for(objProcessorIt pIt = this->objProcessors.begin(); pIt != this->objProcessors.end(); ++pIt){

			(*pIt)->onObjectRemove(obj);
		}

		}
	}

	void removeObject(Object &obj){
		this->removeObject(obj.getName());
	}

	void removeObject(Object *obj){
		assert(obj != NULL);

		this->removeObject(obj->getName());
	}

	Object *getObjByName(std::string name){
		objMapIt it = this->objMap.find(name);

		if(  it == this->objMap.end() ){
			return NULL;
		}
		return it->second;
		
	}

	void addObjectProcessor(objectProcessor *processor){
		this->objProcessors.push_back(processor);
		processor->Init(&this->objMap);
	}

	void removeObjectProcessor(objectProcessor *processor){
		//this->objProcessors.push_back(processor);
	}


	void Process(float dt){
		
		for(objProcessorIt it = this->objProcessors.begin(); it != this->objProcessors.end(); ++it){

			(*it)->Process(dt);
		}
	};
};