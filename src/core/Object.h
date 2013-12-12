#pragma once
#include <map>
#include <vector>
#include "Property.h"
#include "Hash.h"
#include "../util/mathUtil.h"
#include "../util/logObject.h"
#include <memory>

/*!Represents a game Object. 

An Object is a collection of Property objects. it's basically a data carrier. 
every object class has a unique name associated with it. It also has a list of
all Properties owned by itself. An Object is modified upon by objectProcessor classes.
the objectProcessors take the data present in Object and use that for rendering, simulation,
etc. 

It's a component object system. the Object can be thought of as a bag of baseProperty objects, with a unique
key (the unique name). The objectProcessor classes "use" the data present in the Object to simulate
the game

\sa objectProcessor baseProperty 
*/   
class Object{

public:

	/*!Constructs an object given the name
	If a name is repeated twice, then an integer is appended to the name to represent
	the number of times a name has been repeated.

	The first object will be name0. the 2nd object will be name1. The nth object will
	be name(n - 1). 

	@param [in] name the name to which the count integer will be appended to generate a unique name
	*/
	Object(std::string _name);
	~Object();

	/*!returns the unique name of the Object*/
	std::string getName();

	/*!adds a property to the Object
	The baseProperty and Hash is  stored as a key-value pair. The Hash is the
	key.The baseProperty is the value

	@param [in] name The Hash of the Property name
	@param [in] value The Property to be associated with the key name
	*/
	void addProp(const Hash *name, baseProperty *value);
	

	/*!returns the baseProperty associated with the name
	It is not advisable to use this function. it is better to just use
	Object::getProp, Object::getPtrProp, and Object::getManagedProp
	rather than use this function

	@param [in] name The Hash of the name of the property

	\return the Property associated with the name, or NULL if the property does not exist
	*/
	baseProperty *getBaseProp(const Hash *name);


	/*!returns whether the Object has the property with name */
	bool hasProperty(const Hash *name){
		return this->getBaseProp(name) == NULL ? false : true;
	}

	/*!returns the value associated with the name
	This function is used to retrieve a value that was stored in Prop.
	If a value is not attached to name, or if the value is stored in 
	some other derived class member of baseProperty_, the function will
	return NULL. it will _only_ return the value of the Property if both the
	key as well as the derived class type of baseProperty matches

	@param [in] name The Hash of the name of the property
	\return the value associated with the Prop, NULL if the property does not exist,
				NULL if the value is stored in some other derived class member of baseProperty
	*/
	template<typename T>
	T* getProp(const Hash *name){
		Prop<T> *prop =_getProperty<T>(name, false);
		if(prop == NULL){
			return NULL;
		}else{
			return prop->getVal();
		};
	}

	template<typename T>
	void setProp(const Hash *name, T *value){
		assert(name != NULL && value !=  NULL);
		Prop<T> *prop =_getProperty<T>(name);
		prop->setVal(*value);
		
	}

	template<typename T>
	void setProp(const Hash *name, T value){
		Prop<T> *prop =_getProperty<T>(name);
		prop->setVal(value);
	}

	/*!send a one time message to the object, and optionally it's children

	This is used to send a message to the object which is stored. This message
	can be retrieved by the component which is interested and can act based on
	messages. Basically acts as a mini observer pattern within the Object itself

	@param [in] tag: The tag of the message to be sent
	@param [in] value: The value that should go along with the tag. This can be 
						retrieved at the component's site

	\return None
	*/
	template<typename T>
	void sendMessage(const Hash *tag, T value, bool sendToChildren = false) {
		auto it = this->messages.find(tag);

		if(it != this->messages.end()) {
			delete (it->second);	
			this->messages.erase(it);	
		}

		this->messages[tag] = new Prop<T> (value);

		if(sendToChildren){
			for (auto child : children) {
				child->sendMessage<T>(tag, value, sendToChildren);
			}
		}
	}

	void sendMessage(const Hash *tag, bool sendToChildren = false) {
	
		auto it = this->messages.find(tag);

		if(it != this->messages.end()) {
			delete (it->second);
			this->messages.erase(it);	
		}

		//create a stub property 
		this->messages[tag] = new Prop<void>();


		if(sendToChildren){
			for (auto child : children) {
				child->sendMessage(tag, sendToChildren);
			}
		}
	}

	//messages auto-expire once retrieved
	template<typename T=void>
	T* getMessage(const Hash *tag) {
		auto it = this->messages.find(tag);

		if(it == messages.end()) {
			return NULL;
		}

		Prop<T> *prop = dynamic_cast< Prop<T>* >(it->second);

		messages.erase(it);
		return prop->getVal();

		
	};


	void addChild(Object *child){
		children.push_back(child);
	}

	/*!kills the Object. 
	The Object destruction will be notified to all
	objectProcessor classes. It will then be destroyed
	*/
	void Kill(){
		this->dead = true;
	}

	/*!returns whether the Object has died */
	bool isDead(){
		return this->dead;
	}

private:
	friend class objectProcessor;


	//name of the object
	std::string name;


	//vector of all children of this object
	std::vector<Object *>children;

	//a map of properties that can be accessed by all objects
	std::map<const Hash*, baseProperty* > propertyMap;

	//map of messages within the object.
	std::map<const Hash*, baseProperty* > messages;

	//stores the number of times an object of the same name has been created
	//to ensure all names are unique. The way war3 used to do it.
	static std::map<std::string, unsigned int>nameMap;

	//stores whether the Object is dead or not
	bool dead;	

	void _genUniqueName(std::string genericName, std::string &out);
	
	template <typename T>
	Prop<T> * _getProperty(const Hash* name, bool warnIfNull = true){


		Prop<T> *prop = dynamic_cast<Prop<T> *>(this->getBaseProp(name));
		
		if(prop == NULL && warnIfNull){
			util::errorLog<<"unable to find property. \
							\nObject: "<<this->getName()<<"\nProperty: "<<name;
			
			return NULL;
		}

		return prop;
	};
};


typedef std::vector<Object *> objectList;
