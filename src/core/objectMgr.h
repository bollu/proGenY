#pragma once
#include "Object.h"
#include "objectProcessor.h"

#include <vector>
#include "../util/logObject.h"


/*!Manages Object class's lifecycle
the ObjectMgr is in charge of controlling Object and running
objectProcessor on the objects. It's the heart of the component based 
system present in the engine

\sa Object
\sa objectProcessor
\sa objectMgrProc
*/
class objectMgr{

private:
	
	objectMap objMap;

	std::vector<objectProcessor *> objProcessors;
	typedef std::vector<objectProcessor *>::iterator objProcessorIt;
public:
	objectMgr(){};
	~objectMgr(){};

	/*!add an Object to the objectManager for processing
	once an Object is added, it is updated and drawn every frame 
	until the Object dies. 
	
	@param [in] obj the Object to be added
	*/
	void addObject(Object *obj){

		assert(obj != NULL);
		util::infoLog(obj->getName()  + "  added to objectManager");

		this->objMap[obj->getName()] = obj;
		for(objProcessorIt it = this->objProcessors.begin(); it != this->objProcessors.end(); ++it){

			
			(*it)->onObjectAdd(obj);
		}

	}
	
	/*!remove an Object from the objectManager
	this is rarely, if every used. Once an object is removed,
	it will not be processed by objectProcessors
	
	@param [in] name the unique name of the Object
	*/
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

	/*!returns an Object by it's unique name
	@param [in] name the unique name of the Object

	\return the Object with the given name, or NULL
	if the Object does not exist
	*/
	Object *getObjByName(std::string name){
		objMapIt it = this->objMap.find(name);

		if(  it == this->objMap.end() ){
			return NULL;
		}
		return it->second;
		
	}

	/*!add an objectProcessor to make it process Objects
	@param [in] processor the objectProcessor to be added
	*/
	void addObjectProcessor(objectProcessor *processor){
		this->objProcessors.push_back(processor);
		processor->Init(&this->objMap);
	}

	/*!remove an objectProcessor that had been added
	@param [in] processor the objectProcessor to be removed 
	*/
	void removeObjectProcessor(objectProcessor *processor){
		//this->objProcessors.push_back(processor);
	}

	/*!pre-processing takes place
	generally, variables are setup if need be in this step, 
	and everything is made ready for Process
	*/
	void preProcess(){
		for(objectProcessor* processor : objProcessors){
			processor->preProcess();
		}
	}

	/*!cleaning up takes place
	once Process is called, postProcess takes care of cleaning
	behind all of the leftover variables and data.
	Dead Objects are also destroyed in this step
	*/
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

	/*!Processes all objects*/
	void Process(float dt){
		
		for(objectProcessor* processor : objProcessors){
			processor->Process(dt);
		}
	};
};