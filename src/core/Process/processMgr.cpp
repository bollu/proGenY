#pragma once
#include "processMgr.h"
#include "Process.h"
#include <typeinfo>


void processMgr::addProcess(Process *p){

	if(this->processes.find(p->getNameHash()) != this->processes.end()){
		util::errorLog<<"clash of names. 2 processes have the same name.\n Process name: "<<
		p->getNameHash()<<util::flush;
	}

	this->processes[p->getNameHash()] = p;
};

void processMgr::preUpdate(){
	for(auto it = this->processes.begin(); it != this->processes.end(); ++it){
		it->second->preUpdate();
	}
}

void processMgr::Update(float dt){
	for(auto it = this->processes.begin(); it != this->processes.end(); ++it){
		//util::msgLog("Update: " + std::string(typeid(it->second).name()) );
				it->second->Update(dt);
	}
};

void processMgr::Draw(){
	for(auto it = this->processes.begin(); it != this->processes.end(); ++it){
		it->second->Draw();
	}

};

void processMgr::postDraw(){
	for(auto it = this->processes.begin(); it != this->processes.end(); ++it){
		it->second->postDraw();
	}
};


void processMgr::Shutdown(){
	for(auto it = this->processes.begin(); it != this->processes.end(); ++it){
		it->second->Shutdown();
	}
};

void processMgr::PauseProcess(const Hash* processName){

	auto it = this->processes.find(processName);
	if( it != processes.end()){

		it->second->Pause();
		return;
	}
	util::errorLog<<"unable to find process to Pause.\nProcess Name: "<<processName<<util::flush;

}

void processMgr::ResumePorcess(const Hash* processName){

	auto it = this->processes.find(processName);
	if( it != processes.end()){

		it->second->Resume(); 
		return;
	}
	util::errorLog<<"unable to find process to Resume.\nProcess Name: "<<processName<<util::flush;
}

Process *processMgr::_getProcess(const Hash* processName){
	auto it = this->processes.find(processName);
	
	if( it != processes.end()){
		return it->second;
	};
	
	util::errorLog<<"trying to get a process that does not exist.\nProcess name: "<<
				processName<<util::flush;

	return NULL;
};
