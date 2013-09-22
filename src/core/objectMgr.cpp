#pragma once
#include "objectMgr.h"



void objectMgr::addObject(Object *obj){

	assert(obj != NULL);

	std::string name = obj->getName();

	assert(this->objMap.find(name) == this->objMap.end());
	assert(this->activeObjects.find(name) == this->activeObjects.end());

	this->objMap[name] = obj;
	this->activeObjects[name] = obj;


	for(objProcessorIt it = this->objProcessors.begin(); it != this->objProcessors.end(); ++it){
		(*it)->onObjectAdd(obj);
	}
	util::infoLog<<"\n"<<obj->getName()<<"  added to objectManager";

}


void objectMgr::deactivateObject(Object &obj){
	this->deactivateObject(obj.getName().c_str());
};

void objectMgr::deactivateObject(const char* name){
	auto it = this->activeObjects.find(std::string(name));

	util::infoLog<<"\n\ndeactivating "<<name;

	assert(it != this->activeObjects.end());

	Object *obj = it->second; 

	for(objectProcessor* processor : objProcessors){
		processor->onObjectDeactivate(obj);
	}

	this->activeObjects.erase(it);

};

void objectMgr::activateObject(Object &obj){
	this->activateObject(obj.getName().c_str());

};

void objectMgr::activateObject(const char* name){

	auto it = this->objMap.find(std::string(name));
	assert(it != this->objMap.end());

	Object *obj = it->second;

	auto activeIt = this->activeObjects.find(std::string(name));


		//this object is ALREADY in the "active" list
	if(activeIt != this->activeObjects.end()){
		util::warningLog<<"\n\nWARNING: trying to activate "<<name<<". It is already active ";
		return;
	}

	this->activeObjects[std::string(name)] = obj;

	for(objectProcessor* processor : objProcessors){
		processor->onObjectActivate(obj);
	}

}


Object *objectMgr::getObjByName(const char* name){
	Object::objMapIt it = this->objMap.find(name);

	if(it == this->objMap.end()){
		return NULL;
	}
	return it->second;

}

void objectMgr::addObjectProcessor(objectProcessor *processor){
	this->objProcessors.push_back(processor);
	processor->Init(&this->activeObjects);
}


void objectMgr::removeObjectProcessor(objectProcessor *processor){
		//this->objProcessors.push_back(processor);
}

void objectMgr::preProcess(){
	for(objectProcessor* processor : objProcessors){
		processor->preProcess();
	}
}


void objectMgr::postProcess(){
	for(objectProcessor* processor : objProcessors){
		processor->postProcess();
	}

	
	for(auto it = this->activeObjects.begin(); it != this->activeObjects.end(); ){
		Object *obj = it->second;
		std::string name = obj->getName();

		if(obj->isDead()){
			objMap.erase(name);
			it = activeObjects.erase(it);

			for(objectProcessor* processor : objProcessors){
				processor->onObjectDeath(obj);
			}

			util::infoLog<<"\ndestroyed Object "<<name;

		}else{
			++it;
		}

	};
}


void objectMgr::Process(float dt){
	
	for(objectProcessor* processor : objProcessors){
		processor->Process(dt);
	}
};
