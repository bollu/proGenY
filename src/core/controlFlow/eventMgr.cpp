#pragma once
#include "eventMgr.h"
#include "../IO/logObject.h"
#include "../IO/strHelper.h"

eventMgr::eventMgr(){

};

void eventMgr::Register(const Hash *eventName, Observer *observer){
	this->observerMap[eventName].push_back(observer);
};

void eventMgr::Unregister(const Hash *eventName, Observer *observer){
	auto mapIt = this->observerMap.find(eventName);

	if(mapIt != this->observerMap.end()){
		observerList list = mapIt->second;		

		for(auto listIt = list.begin(); listIt != list.end(); ++listIt){
			if(*listIt == observer){

				list.erase(listIt);
				return;

			}
		}
	}

	IO::errorLog<<"unable to find required observer under eventName \
	to remove observer. \nEventName: "<<eventName<<IO::flush;

};




bool eventMgr::_observersPresent(const Hash *eventName){
	return  this->observerMap.find(eventName) != this->observerMap.end();
};

void eventMgr::_sendEvent(Event &event){

	const Hash *currentEventName = event.name;
	baseProperty *currentData = event.data;

	auto mapIt = this->observerMap.find(currentEventName);
	assert(mapIt != this->observerMap.end());
	
	observerList list = mapIt->second;
	for(auto listIt = list.begin(); listIt != list.end(); ++listIt){
		(*listIt)->recieveEvent(currentEventName, currentData);
	}
	
	if(currentData != NULL){
		delete(currentData);
	}
};

void eventMgr::_Dispatch(Event &newEvent){
	
	this->events.push(newEvent);

	//there already is an event that's running. so just include this one
	//in the queue and return. 
	if(this->events.size() > 1){
		return;
	}

	while(!this->events.empty()){
		Event &currentEvent = this->events.front();

		this->_sendEvent(currentEvent);

		this->events.pop();
	};
	//there are no pending events. just send the event
	
	
};


