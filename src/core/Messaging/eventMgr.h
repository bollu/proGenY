#pragma once
#include "../Hash.h"
#include "eventData.h"
#include "../Property.h"
#include <map>
#include <vector>



class Observer{
public:
	virtual void recieveEvent(const Hash *eventName, baseProperty *eventData) = 0;
};



class eventMgr{

public:
	void Register(const Hash *eventName, Observer *observer);
	void Unregister(const Hash *eventName, Observer *observer);

	void sendEvent(const Hash *eventName, baseProperty *eventData = NULL);

private:
	typedef std::vector<Observer *> observerList;

	std::map<const Hash*, observerList>observerMap;


};