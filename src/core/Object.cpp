#pragma once
#include "Object.h"
#include <sstream>
#include "../util/logObject.h"

std::map<std::string, unsigned int> Object::nameMap;

Object::Object(std::string _name){
	//create a unique name from the generic name given
	this->_genUniqueName(_name, this->name);

	this->addProperty(Hash::getHash("position"), new Prop <vector2>(vector2(0, 0)));

};

Object::~Object(){
	for(auto it = this->propertyMap.begin(); it != this->propertyMap.end(); propertyMap.erase(it++)){
		delete ((*it).second);	
	}

}


//public-------------------------------------------
void Object::addProperty(const Hash *name, baseProperty *value){
	propertyIt it;
	
	if( ( it = propertyMap.find(name)) == propertyMap.end() ){
		propertyMap[name] = value;
		
		return;
	}
	util::msgLog("trying to add a property twice to object " + this->getName(), 
		util::logLevel::logLevelError);
}

baseProperty *Object::getBaseProperty(const Hash *name){

	propertyIt it;

	if( (it = propertyMap.find(name)) != propertyMap.end() ){
		return it->second;
	}
	return NULL;
}

std::string Object::getName(){
	return this->name;
};



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

		//it's our responsibilty to increment it
		this->nameMap[genericName]++;

		sstm << genericName << this->nameMap[genericName];
		out = sstm.str();
	}
}