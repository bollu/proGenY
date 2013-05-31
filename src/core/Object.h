#pragma once
#include <map>
#include <vector>
#include "Property.h"
#include "Hash.h"


class Object{

public:

	Object(std::string _name);
	~Object();


	std::string getName();

	void addProp(const Hash *name, baseProperty *value);
	

	baseProperty *getBaseProp(const Hash *name);

	template<typename Type>
	Prop<Type> *getProp(const Hash *name){
		return reinterpret_cast<Prop<Type> *>(this->getBaseProp(name));
	}

	template<typename Type>
	managedProp<Type> *getManagedProp(const Hash *name){
		return reinterpret_cast<managedProp<Type> *>(this->getBaseProp(name));
	}
	

	/*
	//not to be used publicly. ONLY FOR OBJECT PROCESSORS
	void _addData(const Hash *name, baseProperty*);

	//not to be used publicly. ONLY FOR OBJECT PROCESSORS
	baseProperty* _getData(const Hash *name);
	*/

private:
	friend class objectProcessor;


	//name of the object
	std::string name;

	//a map of properties that can be accessed by all objects
	std::map<const Hash*, baseProperty* > propertyMap; 
	typedef std::map<const Hash*, baseProperty* >::iterator propertyIt;


	/*	
	//a map of data that is used *ONLY* by data processors
	std::map<const Hash*, baseProperty*>dataMap;
	typedef std::map<const Hash*, baseProperty*>::iterator dataIt;
	*/

	//stores the number of times an object of the same name has been created
	//to ensure all names are unique. The way war3 used to do it.
	static std::map<std::string, unsigned int>nameMap;
	typedef std::map<std::string, unsigned int>::iterator nameIt;

	void _genUniqueName(std::string genericName, std::string &out);
	
};


typedef std::vector<Object *> objectList;
