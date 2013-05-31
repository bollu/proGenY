#pragma once
#include "processMgr.h"
#include "Process.h"


void processMgr::addProcess(Process *p){

	if(this->processes.find(p->getNameHash()) != this->processes.end()){
		util::msgLog("clash of names. 2 processes have the same name.\n Process name: " +
			Hash::Hash2Str(p->getNameHash()), util::logLevel::logLevelError);
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
	util::msgLog("unable to find process to Pause.\nProcess Name: " + Hash::Hash2Str(processName), 
		util::logLevel::logLevelError);

}

void processMgr::ResumePorcess(const Hash* processName){

	auto it = this->processes.find(processName);
	if( it != processes.end()){

		it->second->Resume(); 
		return;
	}
	util::msgLog("unable to find process to Resume.\nProcess Name: " + Hash::Hash2Str(processName), 
		util::logLevel::logLevelError);
}

Process *processMgr::_getProcess(const Hash* processName){
	auto it = this->processes.find(processName);
	
	if( it != processes.end()){
		return it->second;
	};
	std::cout<<"Unable to find process"<<std::endl;
	return NULL;
};