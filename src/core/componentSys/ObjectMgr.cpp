#pragma once
#include "ObjectMgr.h"



void ObjectMgr::addObject(Object *obj){

	assert(obj != NULL);

	std::string name = obj->getName();

	assert(this->objMap.find(name) == this->objMap.end());
	assert(this->activeObjects.find(name) == this->activeObjects.end());

	this->objMap[name] = obj;
	this->activeObjects[name] = obj;


	for(objProcessorIt it = this->objProcessors.begin(); it != this->objProcessors.end(); ++it){
		(*it)->onObjectAdd(obj);
	}
	IO::infoLog<<"\n"<<obj->getName()<<"  added to objectManager";

}


void ObjectMgr::deactivateObject(Object &obj){
	this->deactivateObject(obj.getName().c_str());
};

void ObjectMgr::deactivateObject(const char* name){
	auto it = this->activeObjects.find(std::string(name));

	IO::infoLog<<"\n\ndeactivating "<<name;

	assert(it != this->activeObjects.end());

	Object *obj = it->second; 

	for(ObjectProcessor* processor : objProcessors){
		processor->onObjectDeactivate(obj);
	}

	this->activeObjects.erase(it);

};

void ObjectMgr::activateObject(Object &obj){
	this->activateObject(obj.getName().c_str());

};

void ObjectMgr::activateObject(const char* name){

	auto it = this->objMap.find(std::string(name));
	assert(it != this->objMap.end());

	Object *obj = it->second;

	auto activeIt = this->activeObjects.find(std::string(name));


		//this object is ALREADY in the "active" list
	if(activeIt != this->activeObjects.end()){
		IO::warningLog<<"\n\nWARNING: trying to activate "<<name<<". It is already active ";
		return;
	}

	this->activeObjects[std::string(name)] = obj;

	for(ObjectProcessor* processor : objProcessors){
		processor->onObjectActivate(obj);
	}

}


Object *ObjectMgr::getObjByName(const char* name){
	Object::objMapIt it = this->objMap.find(name);

	if(it == this->objMap.end()){
		return NULL;
	}
	return it->second;

}

void ObjectMgr::addObjectProcessor(ObjectProcessor *processor){
	this->objProcessors.push_back(processor);
	processor->Init(&this->activeObjects);
}


void ObjectMgr::removeObjectProcessor(ObjectProcessor *processor){
		//this->objProcessors.push_back(processor);
}

void ObjectMgr::preProcess(){
	for(ObjectProcessor* processor : objProcessors){
		processor->preProcess();
	}
}


void ObjectMgr::postProcess(){
	for(ObjectProcessor* processor : objProcessors){
		processor->postProcess();
	}

	
	for(auto it = this->activeObjects.begin(); it != this->activeObjects.end(); ){
		Object *obj = it->second;
		std::string name = obj->getName();

		if(obj->isDead()){
			objMap.erase(name);
			it = activeObjects.erase(it);

			for(ObjectProcessor* processor : objProcessors){
				processor->onObjectDeath(obj);
			}

			IO::infoLog<<"\ndestroyed Object "<<name;

		}else{
			++it;
		}

	};
}


void ObjectMgr::Process(float dt){
	
	for(ObjectProcessor* processor : objProcessors){
		processor->Process(dt);
	}
};
