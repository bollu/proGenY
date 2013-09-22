#pragma once
#include "Object.h"
#include <sstream>
#include "../util/logObject.h"
#include "../util/mathUtil.h"

std::map<std::string, unsigned int> Object::nameMap;

Object::Object(std::string _name) : dead(false), baseName(_name){
	//create a unique name from the generic name given
	this->_genUniqueName(_name, this->name);

	this->addProp(Hash::getHash("position"), new Prop <vector2>(vector2(0, 0)));
	this->addProp(Hash::getHash("facing"), new Prop <util::Angle>(util::Angle::Deg(0)));

};

Object::~Object(){
	for(auto it = this->propertyMap.begin(); it != this->propertyMap.end(); propertyMap.erase(it++)){
		delete ((*it).second);	
	}

}


//public------------------------------------------------------------
//getters-----------------------------------------------------------
std::string Object::getBaseName() const{
	return this->baseName;
}

std::string Object::getName() const{
	return this->name;
};


bool Object::hasProperty(const Hash *name) const{
	return this->_getBaseProp(name) == NULL ? false : true;
}

bool Object::hasProperty(const char *name) const{
	return this->hasProperty(Hash::getHash(name));
}

bool Object::requireProperty(const Hash *name) const{
	if(!this->hasProperty(name)){
		util::errorLog<<"Object "<<this->getName()<<
		" does not have Property "<<name<<" as required"<<util::flush;

		return false;
	}

	return true;
};

bool Object::requireProperty(const char *name) const{
	return this->requireProperty(Hash::getHash(name));
};

bool Object::isDead() const{
	return this->dead;
}

//setters-----------------------------------------------------------
void Object::addProp(const Hash *name, baseProperty *value){
	propertyIt it;
	
	if( ( it = propertyMap.find(name)) == propertyMap.end() ){
		propertyMap[name] = value;
		
		return;
	}

	util::errorLog<<"trying to add property twice to object\n  \
	\nProperty: "<<name<<
	"\nObject Name: "<<this->getName()<<util::flush;
}

void Object::addProp(const char *name, baseProperty *value){
	this->addProp(Hash::getHash(name), value);
};


baseProperty *Object::_getBaseProp(const Hash *name) const{

	cPropertyIt it = propertyMap.find(name);


	if( it  != propertyMap.end() ){
		return it->second;
	}
	return NULL;
}


void Object::_printProperties() const{
	util::infoLog<<"\n\n"<<"name: "<<this->name<<"\n";

	for(auto it =  propertyMap.begin(); it != propertyMap.end(); ++it){
		util::infoLog<<it->first<<"\n";
	}		
}


void Object::Kill(){
	this->dead = true;
}




//private-----------------------------------------------------------

/*
//privates---------------------------------------------
void Object::_addData(const Hash *name, baseProperty* value){
	dataMap[name] = value;
}

baseProperty* Object::_getData(const Hash *name){

	dataIt it;
	if( (it = dataMap.find(name)) != dataMap.end() ){
		return it->second;
	}
	return NULL;
}
*/


void Object::_genUniqueName(std::string genericName, std::string &out){
	nameIt it = this->nameMap.find(genericName);
	std::stringstream sstm;

	//the name hasn't been stored
	if(it == this->nameMap.end()){
		
		this->nameMap[genericName] = 0;


		sstm << genericName << 0;
		out = sstm.str();
	}else{

		//it's our responsibility to increment it
		this->nameMap[genericName]++;

		sstm << genericName << this->nameMap[genericName];
		out = sstm.str();
	}
}
