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

			for(objectProcessor* processor : objProcessors){
				processor->onObjectRemove(obj);
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

	void preProcess(){
		for(objectProcessor* processor : objProcessors){
			processor->preProcess();
		}
	}

	void postProcess(){
		for(objectProcessor* processor : objProcessors){
			processor->postProcess();
		}

		
		for(auto it = this->objMap.begin(); it != this->objMap.end(); ){
			Object *obj = it->second;

			if(obj->isDead()){
				it = objMap.erase(it);

				for(objectProcessor* processor : objProcessors){
					processor->onObjectRemove(obj);
				}

				delete(obj);
				obj = NULL;
			}else{
				it++;
			}
		}
		
	}

	void Process(float dt){
		
		for(objectProcessor* processor : objProcessors){
			processor->Process(dt);
		}
	};
};