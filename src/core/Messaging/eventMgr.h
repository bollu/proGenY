#pragma once
#include "../Hash.h"
#include "eventData.h"
#include "../Property.h"
#include "../../util/logObject.h"
#include "../../util/strHelper.h"
#include <map>
#include <vector>
#include <queue>
#include <list>

/*! Used to watch for events with the eventMgr 

It is an implementation of the Observer pattern. the observer receives the events
that it has been watching for, along with some data sent by the caller. 

The observer pattern is useful to separate various subsystems. it reduces direct
coupling between parts of the engine. Rather, the eventMgr acts as a mediator between
different parts of the engine

\sa eventMgr
*/
class Observer{
public:
	/*! Receive a particular event that has been sent to the eventMgr
	
	@param [in] eventName the name of the event
	@param [in] eventData  data sent by the class that generated the event.
	*/
	virtual void recieveEvent(const Hash *eventName, baseProperty *eventData) = 0;
};


/*! Used for communication between different subsystems 

It is an implementation of the classic observer pattern. 
The eventManager acts as an intermediate between multiple subsystems,
thus reducing coupling between classes in the engine. The event Manager is the focal point
of communication. all messages are sent and received using the eventManager. 

an Observer can register itself to receive messages with the eventManager.
Then, when a message is sent, if an Observer has registered itself with that
particular message, the message is passed to the Observer. 

\sa Observer
*/
class eventMgr{

public:

	eventMgr();

	/*! Used to register an Observer with the eventMgr
	
	@param [in] eventName the Hash of the event that the Observer wants to observer
		The Hash can be generated by calling Hash::getHash

	@param [in] observer the Observer that wishes the observe the particular event
	*/
	void Register(const Hash *eventName, Observer *observer);

	/*!Used to un register an Observer with the eventMgr

	@param [in] eventName the Hash of the event which the Observer wants to stop observing
	@param [in] observer The observer that wishes to stop observing the particular event
	*/
	void Unregister(const Hash *eventName, Observer *observer);


	/*!Used to send a particular event to all Observers
	All Observers registered to eventName will have the message sent to them, along with the
	optional data that the sender is free to include 

	@param[in] eventName the Hash of the event to the broad casted to all Observers
	@param[in] eventData optional data that can be sent along with the event
	*/
	template<class T>
	void sendEvent(const Hash *eventName, T &eventData){

		if(! _observersPresent(eventName)){

			util::msgLog("no subscribers to event.\nEventName: " \
				+ Hash::Hash2Str(eventName), util::logLevel::logLevelWarning);

			return;
		}

		Event e;
		e.name = eventName;
		e.data = new Prop<T>(eventData);

		this->_Dispatch(e);

	}

	void sendEvent(const Hash *eventName){
		
		if(! _observersPresent(eventName)){

			util::msgLog("no subscribers to event.\nEventName: " \
				+ Hash::Hash2Str(eventName), util::logLevel::logLevelWarning);

			return;
		}

		Event e;
		e.name = eventName;
		e.data = NULL;

		this->_Dispatch(e);
	}

	


private:

	struct Event{
		const Hash *name;
		baseProperty *data;
	};

	
	bool _observersPresent(const Hash *eventName);
	void _Dispatch(Event &newEvent);
	void _sendEvent(Event &event);


	typedef std::vector<Observer *> observerList;

	std::map<const Hash*, observerList>observerMap;

	std::queue <Event>events;

};