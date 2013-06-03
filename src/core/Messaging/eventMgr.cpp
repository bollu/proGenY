#pragma once
#include "eventMgr.h"
#include "../../util/logObject.h"



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

	util::msgLog("unable to find required observer under eventName to remove observer. \nEventName: " \
		+ Hash::Hash2Str(eventName), util::logLevel::logLevelError);

};

void eventMgr::sendEvent(const Hash *eventName, baseProperty *eventData){
	auto mapIt = this->observerMap.find(eventName);

	if(mapIt == this->observerMap.end()){
		//util::msgLog("no subscribers to event.\nEventName: " \
		//		+ Hash::Hash2Str(eventName), util::logLevel::logLevelWarning);

		return;
	}

	observerList list = mapIt->second;
	for(auto listIt = list.begin(); listIt != list.end(); ++listIt){
		(*listIt)->recieveEvent(eventName, eventData);
	}


};


