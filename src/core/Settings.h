#pragma once
#include "Property.h"
#include "../util/logObject.h"



class Settings{
public:
	Settings(){};
	void loadSettingsFromFile(std::string filePath){};

	template<typename T>
	T* getProp(const Hash *propertyName){
		auto it = settingsMap.find(propertyName);

		if( it == settingsMap.end() ){
				util::msgLog("trying to get a setting that does not exist.\nSetting Name: " + 
					Hash::Hash2Str(propertyName), util::logLevel::logLevelError);
		};

		Prop<T> *prop = dynamic_cast< Prop<T>* >(it->second); 

		if(prop == NULL){
			util::msgLog("trying to get a setting of the wrong type.\nSetting Name: " + 
					Hash::Hash2Str(propertyName),
					util::logLevel::logLevelError) ;
		}
		return prop->getVal();
	};
	


	void addProp(const Hash *propertyName, baseProperty* property){

		if(this->settingsMap.find(propertyName) != settingsMap.end()){
			util::msgLog("trying to add a setting twice.\nSetting Name:" +
			 Hash::Hash2Str(propertyName), util::logLevel::logLevelError);
		}

		settingsMap[propertyName] = property;
	}
	
private:
	std::map<const Hash *, baseProperty*>settingsMap; 

};