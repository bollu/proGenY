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
	template<typename Type>
	Type* getProp(const Hash *name){
		Prop<Type> *prop =_getProperty<Type>(name, false);
		if(prop == NULL){
			return NULL;
		}else{
			return prop->getVal();
		};
	}

	template<typename Type>
	void setProp(const Hash *name, Type *value){
		assert(name != NULL && value !=  NULL);
		Prop<Type> *prop =_getProperty<Type>(name);
		prop->setVal(*value);
		
	}
	
	template<typename Type>
	Prop<Type> *getPropPtr(const Hash *name){
		Prop<Type> *prop = dynamic_cast<Prop<Type> *>(this->getBaseProp(name));

		if(prop == NULL){
			util::errorLog<<"unable to find property. \
							\nObject: "<<this->getName()<<"\nProperty: "<<name;
		}else{
			return prop;
		}
	}

	template<typename Type>
	void setProp(const Hash *name, Type value){
		Prop<Type> *prop =_getProperty<Type>(name);
		prop->setVal(value);
	}


	/*!returns whether the Object has the property with name */
	bool hasProperty(const Hash *name){
		return this->getBaseProp(name) == NULL ? false : true;
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

	//a map of properties that can be accessed by all objects
	std::map<const Hash*, baseProperty* > propertyMap; 
	typedef std::map<const Hash*, baseProperty* >::iterator propertyIt;



	//stores the number of times an object of the same name has been created
	//to ensure all names are unique. The way war3 used to do it.
	static std::map<std::string, unsigned int>nameMap;
	typedef std::map<std::string, unsigned int>::iterator nameIt;


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
